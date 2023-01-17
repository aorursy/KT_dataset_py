import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import matplotlib







df = pd.read_csv('../input/diabetes.csv')

df.head()
df.info()
# There are no null values in the columns

df.isnull().sum()
plt.style.use('ggplot')



def plot_attributes(train,test,features,lable1,lable2,plt,i,j):

    alpha = 0.6

    ax1 = plt.subplot2grid((2, 4), (i, j))

    train[features].plot(kind='kde', color='#FA2379', label=lable1, alpha=alpha)

    test[features].plot(kind='kde', label=lable2, alpha=alpha)

    ax1.set_xlabel(features)

    ax1.set_title("What's the distribution of "+features+"?",fontsize=8)

    plt.legend(loc=1,fontsize=8)

    plt.tight_layout()

    

def plot_KDE(diabetics,nondiabetics,col_names):

    #plt.rc('font', size=13)

    fig = plt.figure(figsize=(18, 8))



    j = 0

    i = 0



    for features in col_names:

        plot_attributes(diabetics,nondiabetics,features,"diabetics","non_diabetics",plt,i,j)

        j = j + 1

        if j == 4:

            j = 0

            i = 1

    

            

col_names= list(df.columns.values)

col_names.remove('Outcome')



diabetics=df[df['Outcome']==1]

nondiabetics=df[df['Outcome']==0]



plot_KDE(diabetics, nondiabetics, col_names)


alpha=0.2

fig = plt.figure(figsize=(6, 6))

ax1 = fig.add_subplot(111)

diabetics['Pregnancies'].value_counts().sort_index().plot(kind='bar', color='#FA2379', alpha=alpha,label="diabetics")

nondiabetics['Pregnancies'].value_counts().sort_index().plot(kind='bar', color='#23FA79', alpha=alpha,

                                                             label="non-diabetics")

ax1.set_ylabel('Frequency')

ax1.set_ylim((0,107))

plt.legend(loc=1,fontsize=8)

ax1.set_title("Number of pregnancy histogram", y=1.05)

plt.grid()

plt.show()

#How are the features correlated?

print(df.corr())
plt.matshow(df.corr())
df.describe()
#Scale the data:

df_norm=df.copy()



features_to_normilize=[ 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction']

from sklearn.preprocessing import MinMaxScaler

df_norm[features_to_normilize] = MinMaxScaler().fit_transform(df_norm[features_to_normilize])


from sklearn.model_selection import train_test_split

X = df_norm.ix[:, df_norm.columns !='Outcome']

y = df_norm.ix[:, -1]

train_data, test_data, y_train, actual_test_prediction = train_test_split(X, y, test_size=0.2, random_state=1)



# https://www.kaggle.com/lejustin/d/uciml/pima-indians-diabetes-database/feature-engineering-metric-comparison





normals = [0]*3

variables = ['Glucose', 'SkinThickness', 'BMI']



# Generate imputation values with Gaussian randomness.

for n, v in zip(range(len(normals)), variables):

    # Shift the mean up to account for skewness caused by zeros.

    v_mean=train_data[v].mean()*1.5



    # Use surrogate deviation.

    # (Sometimes I get strange values when using .std(). Why?)

    v_std = v_mean*0.1



    normals[n] = np.random.normal(loc = v_mean, scale = v_std)



print("Imputing zeros in Glucose, SkinThickness, and BMI with")

print("%f, %f, and %f" % (normals[0], normals[1], normals[2]))



# Impute.

train_data = train_data.replace(to_replace = {'Glucose': {0: normals[0]},

                                  'SkinThickness': {0: normals[1]},

                                  'BMI': {0: normals[2]}})



normals = [0]*3

variables = ['Glucose', 'SkinThickness', 'BMI']



# Generate imputation values with Gaussian randomness.

for n, v in zip(range(len(normals)), variables):

    # Shift the mean up to account for skewness caused by zeros.

    v_mean = test_data[v].mean()*1.5



    # Use surrogate deviation.

    # (Sometimes I get strange values when using .std(). Why?)

    v_std = v_mean*0.1



    normals[n] = np.random.normal(loc = v_mean, scale = v_std)



print("Imputing zeros in Glucose, SkinThickness, and BMI with")

print("%f, %f, and %f" % (normals[0], normals[1], normals[2]))



# Impute.

X_test = test_data.replace(to_replace = {'Glucose': {0: normals[0]},

                                  'SkinThickness': {0: normals[1]},

                                  'BMI': {0: normals[2]}})



print("DONE")
def calculate_prediction_scores(TP,FN,FP,TN):



    # print TP,FN,FP,TN

    precision=None

    recall=None

    result_dic=dict()



    # recall (TPR)

    try:

        recall=TP/float(TP+FN)

        result_dic['recall']=recall

        result_dic['TPR'] = recall

    except:

        result_dic['TPR'] = 0



    # FPR

    try:

        FPR=FP/float(FP+TN)

        result_dic['FPR']=recall

    except:

        result_dic['FPR'] = 0





    # precision

    try:

        precision = TP / float(TP + FP)

        result_dic['precision']=precision

    except:

        result_dic['precision'] = np.nan



    # f1

    if precision is not None and recall is not None:

        try:

            f1 = (2*precision*recall) / float(precision+recall)

            result_dic['f1'] = f1

        except:

            result_dic['f1'] = np.nan

    else:

        result_dic['f1']=np.nan



    # accuracy

    try:

        accuracy = (TP+TN) / float(TP + FP +TN +FN)

        result_dic['accuracy']=accuracy

    except:

        result_dic['accuracy'] = np.nan





    # MCC

    try:

        top = (TP*TN) - (FP*FN)

        bottom=(TP+FN)*(TP+FP)*(TN+FP)*(TN+FN)

        bottoms= sqrt((bottom))

        mcc=top/float(bottoms)

        result_dic['mcc']=mcc

    except:

        result_dic['mcc'] = np.nan

    return result_dic



# given two binary classiifcation calcuulate tP,FP,..

def calculate_performance_for_binary_classification(actual_binary_list, predict_binary_list):



    TP=0; FN=0; FP=0; TN=0;



    for i in range(0,len(actual_binary_list)):

        if actual_binary_list[i]==1 and predict_binary_list[i]==1:

            TP=TP+1

        if actual_binary_list[i] == 1 and predict_binary_list[i]==0:

            FN = FN + 1

        if actual_binary_list[i] == 0 and predict_binary_list[i] == 0:

            TN = TN + 1

        if actual_binary_list[i] == 0 and predict_binary_list[i] == 1:

            FP = FP+1



    dic_scores=calculate_prediction_scores(TP,FN,FP,TN)

    return dic_scores
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import roc_curve,roc_auc_score,auc



def plot_roc_curve(actual_test_prediction,Predictedprobabilities):



    fpr, tpr, _ = roc_curve(actual_test_prediction, np.array(Predictedprobabilities))

    roc_auc = auc(fpr, tpr)

    plt.figure()

    lw = 2

    plt.plot(fpr, tpr, color='darkorange',

             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)

    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

    plt.xlim([0.0, 1.0])

    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.title('Receiver operating characteristic example')

    plt.legend(loc="lower right")

    plt.show()

    

# Create the random forest object which will include all the parameters

# for the fit

forest = RandomForestClassifier(n_estimators = 100)





# Fit the training data to the Survived labels and create the decision trees

forest_model = forest.fit(train_data,y_train)



# Take the same decision trees and run it on the test data

prediction_array = forest_model.predict(test_data)

prob= forest.predict_proba(test_data)[:,1]

plot_roc_curve(actual_test_prediction,prob)

calculations=calculate_performance_for_binary_classification(np.array(actual_test_prediction),prediction_array)

print(calculations)
def plot_feature_importance(X,y,h):

    from sklearn.ensemble import ExtraTreesClassifier

    # Build a forest and compute the feature importances

    forest = ExtraTreesClassifier(n_estimators=250,

                                  random_state=0)



    forest.fit(X, y)

    importances = forest.feature_importances_

    std = np.std([tree.feature_importances_ for tree in forest.estimators_],axis=0)

    indices = np.argsort(importances)[::-1]



    print("Feature ranking:")

    lables=[]

    for f in range(X.shape[1]):

        l= h[indices[f]]

        lables.append(l)

        print("%d. feature %s (%f)" % (f + 1,l, importances[indices[f]]))



    # Plot the feature importances of the forest

    plt.figure()

    plt.title("Feature importances")

    plt.bar(range(X.shape[1]), importances[indices],

           color="r", yerr=std[indices], align="center")

    plt.xticks(range(X.shape[1]), lables,rotation=90)

    plt.xlim([-1, X.shape[1]])

    plt.show()



col_names= list(df.columns.values)

col_names.remove('Outcome')

plot_feature_importance(X,y,col_names)
test_data2=test_data[['Glucose','Age','BMI']]

train_data2=train_data[['Glucose','Age','BMI']]



# Fit the training data to the Survived labels and create the decision trees

forest_model = forest.fit(train_data2,y_train)



# Take the same decision trees and run it on the test data

prediction_array = forest_model.predict(test_data2)

prob= forest.predict_proba(test_data2)[:,1]

plot_roc_curve(actual_test_prediction,prob)

calculations=calculate_performance_for_binary_classification(np.array(actual_test_prediction),prediction_array)

print(calculations)