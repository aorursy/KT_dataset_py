import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

from sklearn.svm import SVC
from sklearn.preprocessing import label_binarize
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
%matplotlib inline
def parse_row(row):
    row = row.split(".")[0]
    return row

def to_hyperthyroid(row):
    if row != "negative":
        row = "hyperthyroid"
    return row

def to_hypothyroid(row):
    if row != "negative":
        row = "hypothyroid"
    return row

def convert_category(dataframe, column):
    
    if column == 'Sex':
        conditionF = dataframe[column] == 'F' # For sex column
        conditionT = dataframe[column] == 'M' # For sex column
    else:
        conditionF = dataframe[column] == 'f'
        conditionT = dataframe[column] == 't'
    
    dataframe.loc[conditionF, column] = 0
    dataframe.loc[conditionT, column] = 1
columns = ["Age", "Sex", "On Thyroxine", "Query on Thyroxine", 
           "On Antithyroid Medication", "Sick", "Pregnant", 
           "Thyroid Surgery", "I131 Treatment", "Query Hypothyroid", 
           "Query Hyperthyroid", "Lithium", "Goitre", "Tumor", 
           "Hypopituitary", "Psych", "TSH Measured", "TSH", "T3 Measured", 
           "T3", "TT4 Measured", "TT4", "T4U Measured", "T4U", 
           "FTI Measured", "FTI", "TBG Measured", "TBG", "Referral Source", "Category"]
hyper_data = pd.read_csv("../input/adarsh/allhyper.csv", names=columns)
hypo_data = pd.read_csv("../input/adarsh/allhypo.csv", names=columns)
sick_data = pd.read_csv("../input/adarsh/sick.csv", names=columns)
hyper_data['Category'] = hyper_data['Category'] \
                            .apply(parse_row) \
                            .apply(to_hyperthyroid)

hypo_data['Category'] = hypo_data['Category'] \
                            .apply(parse_row) \
                            .apply(to_hypothyroid)

sick_data['Category'] = sick_data['Category'] \
                            .apply(parse_row)

thyroid_frames = [hyper_data, hypo_data, sick_data]
thyroid_data = pd.concat(thyroid_frames) \
                 .drop_duplicates() \
                 .drop(['Referral Source', 'TBG', 'TBG Measured'], axis=1)
classes = thyroid_data['Category'].unique()
print("Number of samples:", len(thyroid_data))
thyroid_data.head()
thyroid_data.loc[thyroid_data['Age'] == '455', 'Age'] = '45'
binary_cols = ['On Thyroxine', 'Query on Thyroxine', 'Sex',
               'On Antithyroid Medication', 'Sick', 'Pregnant', 
               'Thyroid Surgery', 'I131 Treatment', 'Query Hypothyroid', 
               'Query Hyperthyroid', 'Lithium', 'Goitre', 'Tumor', 
               'Hypopituitary', 'Psych', 'TSH Measured', 'T3 Measured', 
               'TT4 Measured', 'T4U Measured', 'FTI Measured']

for col in binary_cols: convert_category(thyroid_data, col)
for col in thyroid_data.columns: 
    if col != 'Category':
        thyroid_data.loc[thyroid_data[col] == '?', col] = np.nan
        thyroid_data[col] = pd.to_numeric(thyroid_data[col])
curr_columns = thyroid_data.columns.difference(['Category'])

imputer = SimpleImputer(missing_values=np.nan, strategy='median')
imputed_data = imputer.fit_transform(thyroid_data.drop('Category', axis=1))
imputed_data = pd.DataFrame(imputed_data, columns=curr_columns)

thyroid_data = pd.concat([
                    imputed_data.reset_index(), 
                    thyroid_data['Category'].reset_index()], 
                    axis=1).drop('index', axis=1)
thyroid_data.head()
X = thyroid_data.drop('Category', axis=1)
y = thyroid_data['Category']

col_names = X.columns

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y)

scaler = MinMaxScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train))
X_test = pd.DataFrame(scaler.transform(X_test))
def plot_pca_data(X, y):    
    
    LE = LabelEncoder()
    y_encoded = LE.fit_transform(y)
    
    pca = PCA(n_components=2)
    pca.fit(X)
    pca_X = pca.transform(X)

    x_axis = pca_X[:,0]
    y_axis = pca_X[:,1]

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    
    for label in np.unique(y):
        ax.scatter(pca_X[y==label, 0],
                   pca_X[y==label, 1], 
                   label=label,
                   s=100,
                   edgecolor='k')
        

    ax.legend()
    plt.xticks([])
    plt.yticks([])

plot_pca_data(X, y)

def search_for_parameters(estimator, X_data, y_data, grid_parameters, scoring='accuracy', cv=3):
    
    classes = y_data.unique()
    
    # Execute Grid Search
    grid_clf = GridSearchCV(estimator=estimator, scoring=scoring,
                            param_grid=grid_parameters, iid=False,
                            cv=cv,n_jobs=-1)
    
    grid_clf.fit(X_data, y_data)

    print("The best parameters are: ", grid_clf.best_params_)

    return grid_clf.best_params_
svm_clf = SVC(class_weight='balanced')

svm_params_list = {'C':[1,2,4,8], 
                   'kernel':['poly', 'rbf', 'sigmoid'], 
                   'degree':[3,4,5], 
                   'gamma':['auto','scale']}

svm_parameters = search_for_parameters(estimator=svm_clf, 
                                X_data=X_train, 
                                y_data=y_train, cv=5,
                                grid_parameters=svm_params_list)

svm_clf.set_params(**svm_parameters)

svm_clf.fit(X_train, y_train)
def cross_validate(estimator, X_data, y_data, scoring='accuracy', cv=3, Z=2):
    
    classes = y_data.unique()
    
    # Execute Cross Validation
    scores = cross_val_score(estimator=estimator, X=X_data, y=y_data, cv=cv, scoring=scoring)

    print("Model Scoring Evaluation Results")
    print("The mean score and the confidence interval of the score estimate are:")
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * Z))

cross_validate(estimator=svm_clf, X_data=X_train, y_data=y_train, cv=5)

y_true, y_pred = y_test, svm_clf.predict(X_test)

print("Accuracy:",accuracy_score(y_test, y_pred))
print("*"*50)
print()
print("\t\t\tClassification Report")
print()
print(classification_report(y_true, y_pred))