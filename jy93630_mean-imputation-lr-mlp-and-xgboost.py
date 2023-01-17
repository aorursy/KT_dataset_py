import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(color_codes=True)
%matplotlib inline

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc
data = pd.read_csv("../input/diabetes.csv")
data.head()
print(data.info())
data.isnull().sum()
sns.set_context("paper")
sns.set(font='serif')
sns.set_style("white", {
    "font.family": "serif",
    "font.serif": ["Times", "Palatino", "serif"]
})
data['Outcome'].value_counts().plot(kind='bar') 
plt.show()
columns = list(data.columns.values)
features = [x for x in columns if x != 'Outcome']
sns.pairplot(data, hue='Outcome', 
             x_vars=features, y_vars=features, height=2.5)
plt.show()
# Percentages of missing values 
imputation_features = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for feature in imputation_features :
    print(feature, round(len(data[data[feature] == 0]) / len(data), 4))
column_means = data[imputation_features].replace(0, np.NaN).mean(axis=0)
for feature in imputation_features : 
    data.loc[data[feature]== 0, feature] = column_means[feature]
data['intercept'] = 1
data_x = data[features]
data_y = data['Outcome']
train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=0.2, random_state = 777)
def draw_roc(y_true, y_probas) : 
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(1):
        fpr[i], tpr[i], _ = roc_curve(y_true, y_probas)
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_probas.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    plt.figure()
    lw = 2
    plt.plot(fpr[0], tpr[0], color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[0])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()
    
    return roc_auc[0]

def cm_to_metric(cm, method) : 
    TN = CM[0][0]
    FN = CM[1][0]
    TP = CM[1][1]
    FP = CM[0][1]
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    # Specificity or true negative rate
    TNR = TN/(TN+FP) 
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    # Negative predictive value
    NPV = TN/(TN+FN)
    # Overall accuracy
    ACC = (TP+TN)/(TP+FP+FN+TN)
    F1 = 2*(PPV*TPR)/(PPV+TPR)
    return pd.DataFrame({'Method':[method], 'Sensitivity':[TPR], 'Specificity':[TNR], 'PPV':[PPV], 'NPV':[NPV],"ACC":[ACC], "F1":[F1]})
logit = sm.Logit(train_y, train_x) 
result = logit.fit()

# Scipy error fixing..
from scipy import stats
stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)
print(result.summary())
print(np.exp(result.params))
y_pred = result.predict(test_x[features])
draw_roc(test_y, y_pred)
y_pred_class = [1 if x >= 0.5 else 0 for x in result.predict(test_x[features])]
CM = confusion_matrix(test_y, y_pred_class)
metric = cm_to_metric(CM, 'LogisticRegression')
metric
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import Adam

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

adam = Adam(lr=0.01)
# define baseline model
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(4, input_dim=8, init='normal', activation='relu'))
    model.add(Dense(4, init='normal', activation='relu'))
    model.add(Dense(1, init='normal', activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    return model
features = [x for x in features if x != 'intercept']
train_x = train_x[features]
test_x =  test_x[features]
model = baseline_model()
history = model.fit(train_x, train_y, epochs=300, batch_size=128, validation_split=0.2)
# list all data in history
print(history.history.keys())

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
y_pred_nn = model.predict(test_x)
draw_roc(test_y, y_pred_nn)
y_pred_class = model.predict_classes(test_x)
CM = confusion_matrix(test_y, y_pred_class)
metric = pd.concat([metric,cm_to_metric(CM, 'MultiLayerPerceptron')])
metric
from sklearn.ensemble.partial_dependence import partial_dependence, plot_partial_dependence
from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier(n_estimators=100, max_depth=4,learning_rate=0.1, loss='deviance',random_state=1)
model.fit(train_x, train_y)
test_y.shape
y_pred_xgboost = model.predict_proba(test_x)[:,1]
draw_roc(test_y, y_pred_xgboost)
y_pred_class = model.predict(test_x)
CM = confusion_matrix(test_y, y_pred_class)
metric = pd.concat([metric,cm_to_metric(CM, 'XGBoost')])
metric
my_plots = plot_partial_dependence(model,features=[0, 1, 2], # column numbers of plots we want to show
                                   X=test_x,            # raw predictors data.
                                   feature_names=['BMI', 'Age', 'Glucose'], # labels on graphs
                                   grid_resolution=10) # number of values to plot on x axis