import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns



import warnings

warnings.filterwarnings('ignore')
dataset = pd.read_csv('../input/clicked-on-add/advertising.csv')
dataset.head()
dataset.keys()
dataset.head()
# removing spaces in column names...since it will get easier during model deployment..

dataset.columns = dataset.columns.str.replace(' ', '_')
dataset.keys()
# checking datatype in each column.



dataset.info()
# Numerical coumn insights.



dataset.describe().transpose()
# checking for null values..



dataset.isna().sum()
# cross checking for null values..Making a list of missing value types



missing_values = ["n/a", "na", "--"," "]

df = pd.read_csv('../input/clicked-on-add/advertising.csv', na_values = missing_values)
df.isna().sum()
dataset.shape
# seaborn pairplot



sns.pairplot(data = dataset, hue = 'Clicked_on_Ad')

plt.show()
# Checking columnswise data visualization.

# taking non object datatype column..



df1 = dataset.select_dtypes(exclude=[np.object])

for i, col in enumerate(df1.columns):

    sns.set_style('whitegrid')

    plt.figure(i)

    fig, ax =plt.subplots(figsize=(10,5))

    sns.set(font_scale = 1.2)

    sns.kdeplot(df1[col], shade=True)

    plt.show()
# correlation matrix.



print(sns.heatmap(dataset.corr()<-0.8, annot=True))
print(sns.heatmap(dataset.corr()<-0.8, annot=True))
# Checking for total no. of unique values in object datatypes..



df2 = dataset.select_dtypes(include=[np.object])

for i in df2.columns:

    unique_value = df2[i].nunique()

    print("total unique values in '{}' : {}".format(i, unique_value))
 # We will drop two columns from main dataset..1) Ad Topic Line & 2) Timestamp. Since this information is not helpful.

    

dataset1 = dataset.drop(['Ad_Topic_Line','Timestamp'], axis=1)
dataset1.head()
# Converting categorical datatyps...



dataset1["Country"] = dataset1.Country.astype('category').cat.codes

dataset1["City"] = dataset1.City.astype('category').cat.codes
dataset1.head()
x = dataset1.drop(['Clicked_on_Ad'], axis=1)
y = dataset1.loc[:,['Clicked_on_Ad']]
x.head()
y.head()
# We will use recursive feature elimination technique(RFE)



from sklearn.feature_selection import RFE

from sklearn.linear_model import LogisticRegression



log_reg = LogisticRegression()

rfe = RFE(log_reg,7)

rfe_fit = rfe.fit(x,y)
rfe_fit.ranking_
len(rfe_fit.ranking_)  # which equals to no. of columns in x dataset
from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, test_size = 0.2, random_state =15)



x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, train_size = 0.9, test_size = 0.1, random_state = 15)



print(x_train.shape)

print(x_test.shape)

print(x_valid.shape)

print()

print(y_train.shape)

print(y_test.shape)

print(y_valid.shape)
list1 = [y_train, y_test, y_valid]

m=1

for i in list1:

    plt.title(m)

    sns.countplot(x='Clicked_on_Ad', data = i)

    plt.show()

    m +=1
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()



parameters = [{'solver':['newton-cg','lbfgs','liblinear','sag','saga']},{'random_state':[10]},

              {'C':[[i for i in np.geomspace(1e-3, 1e1, num=20)]]}]
# using GridSearchCV



from sklearn.model_selection import GridSearchCV



# help(GridSearchCV)

grid_search = GridSearchCV(estimator= model, param_grid= parameters, scoring='accuracy', n_jobs= -1)



grid_search = grid_search.fit(x_train, y_train)
accuracy = grid_search.best_score_

accuracy
estimator = grid_search.best_estimator_

estimator
grid_search.best_params_
from sklearn.linear_model import LogisticRegression



# creating instance.. 

model1 = LogisticRegression(penalty='l2', solver = 'newton-cg', random_state=10)



# fitting model..

model1.fit(x_train, y_train)



# predicting results for y variable using dependent variable set x_train

y_pred = model1.predict(x_train)
# Generating predict values in probabilites % format..

# first value is correspondance to '0' & second value is with '1'



predict_proba1 = model1.predict_proba(x_train).round(2)

predict_proba1
# getting beta coefficients or beta weights.



model1.coef_
# Accuracy on Train



print("The Training Accuracy is: ", model1.score(x_train, y_train))



# Accuracy on Test

print("The Testing Accuracy is: ", model1.score(x_test, y_test))
# classification report..



from sklearn.metrics import classification_report

print(classification_report(y_train, y_pred))
# confusion matrix..



from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_train, y_pred)

cm
# writing function to plot confusion matrix:



def plot_confusion_matrix(cm, classes=None, title='Confusion matrix'):

    

    """Plots a confusion matrix."""

    if classes is not None:

        sns.heatmap(cm, cmap="YlGnBu", xticklabels=classes, yticklabels=classes, vmin=0., vmax=1., 

                    annot=True, annot_kws={'size':30})

    else:

        sns.heatmap(cm, vmin=0., vmax=1.)

    plt.title(title)

    plt.ylabel('True label')

    plt.xlabel('Predicted label')
# plotting confusion-matrix % value



cm = confusion_matrix(y_train, y_pred)

cm_norm = cm / cm.sum(axis=1)



plot_confusion_matrix(cm_norm, classes = model1.classes_, title='Confusion matrix')


from sklearn.metrics import precision_recall_curve

from sklearn.metrics import plot_precision_recall_curve

import matplotlib.pyplot as plt



# for training datasets.. since we used x_train to find y_pred.

disp = plot_precision_recall_curve(model1, x_train, y_train)

disp.ax_.set_title('Precision/recall tradeoff (training dataset)')

plt.show()



# for testing datasets.. since we used x_train to find y_pred.

disp = plot_precision_recall_curve(model1, x_test, y_test)

disp.ax_.set_title('Precision/recall tradeoff (testing dataset)')

plt.show()
# easy way to plot



# from sklearn.metrics import plot_roc_curve

# disp1 = plot_roc_curve(model1, x_train, y_train)

# plt.show()
# writing function to reuse..key 

def roc(y_model,y_predicted):

    from sklearn.metrics import roc_curve, roc_auc_score

    fpr, tpr, thresholds = roc_curve(y_model,y_predicted)

    auc = roc_auc_score(y_model,y_predicted)

    plt.figure(figsize=(8,6))

    plt.plot(fpr, tpr, linewidth=2, label='Logistic Regression(area = %0.2f)'%auc)

    plt.plot([0, 1], [0, 1], "k--")

    plt.axis([0, 1, 0, 1])

    plt.legend(loc='lower right')

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.title(label= 'Roc Curve')
print('roc curve with training dataset')

roc(y_train, y_pred)
y_predict_t = model1.predict(x_test)

roc(y_test, y_predict_t)
from sklearn.metrics import log_loss



# Running Log loss on training

# predict_proba1 = model1.predict_proba(x_train)  # we have already done this step no.5

print("The Log Loss on Training is: ", log_loss(y_train, predict_proba1))



# Running Log loss on testing set

pred_proba_t = model1.predict_proba(x_test)

print("The Log Loss on Testing Dataset is: ", log_loss(y_test, pred_proba_t))
# we will find out best c value which will give min. log loss & better accuracy..



C_list = np.geomspace(1e-5, 1e2, num=20) #  log space



from sklearn.linear_model import LogisticRegressionCV 

model2 = LogisticRegressionCV(random_state=10, solver='newton-cg', Cs= C_list)

model2.fit(x_train, y_train)



print("The accuaracy is:", model2.score(x_test, y_test))

pred_proba_t = model2.predict_proba(x_test)

log_loss2 = log_loss(y_test, pred_proba_t)

print("The Logistic Loss is: ", log_loss2)



print("The optimal C parameter is: ", model2.C_)
from sklearn.dummy import DummyClassifier



# class sklearn.dummy.DummyClassifier(*, strategy='warn', random_state=None, constant=None) ...fyi only



strategies = ['stratified','most_frequent','prior', 'uniform', 'constant']

test_score = []



for s in strategies:

    if s == 'constant':

        dummy_clf = DummyClassifier(strategy=s, random_state=15, constant = 'break')

    else:

        dummy_clf = DummyClassifier(strategy=s, random_state=15)

        dummy_clf.fit(x_train, y_train)

        score = dummy_clf.score(x_test, y_test)

        test_score.append(score)



        pred_proba_t = dummy_clf.predict_proba(x_test)

        log_loss2 = log_loss(y_test, pred_proba_t)

        

        print("when strategy is '{}'".format(s))

        print("the Testing Acc:", score)

        print("the Log Loss:", log_loss2)

        print('-'*30)
dummy_score = pd.DataFrame(list(zip(strategies, test_score)), columns =['strategies', 'test_score'])
plt.figure(figsize=(8,6))

ax = sns.stripplot(x="strategies", y="test_score", linewidth=3,data=dummy_score)
from sklearn.linear_model import LogisticRegression



final_model = LogisticRegression(penalty='l2', solver = 'newton-cg', random_state=10, C=0.00162378)

final_model.fit(x_train, y_train)

final_predict = final_model.predict(x_valid)



score = final_model.score(x_valid, y_valid)



pred_proba_t3 = final_model.predict_proba(x_valid)

log_loss3 = log_loss(y_valid, pred_proba_t3)



print("Testing Acc:", score)

print("Log Loss:", log_loss3)
final_model
from sklearn.metrics import confusion_matrix

confusion_matrix(y_valid, final_predict)
# we have created function to take user input values..Copy-paste this code in streamlit python file.



def inputs(Daily_Time_Spent_on_Site, Age, Area_Income,Daily_Internet_Usage, City, Male, Country):

    

    new_data=pd.DataFrame({'Daily_Time_Spent_on_Site':Daily_Time_Spent_on_Site,"Age":Age,"Area_Income":Area_Income, "Daily_Internet_Usage":Daily_Internet_Usage,

                           "City":City, "Male":Male, "Country":Country},index=[1])

    

    new_data[[" Daily_Time_Spent_on_Site","Area_Income","Daily_Internet_Usage"]] = df[[" Daily_Time_Spent_on_Site","Area_Income","Daily_Internet_Usage"]].astype('float')

    

    new_data[["Age","City","Male","Country"]] = df[["Age","City","Male","Country"]].astype('int')

    final_predict = final_model.predict(new_data)

    return(final_predict).values
# categorical codes & label dictonary to use in deployment



# for city column.



labels1 = dataset['City'].astype('category').cat.categories.tolist()

city_codes = {k: v for k,v in zip(labels1,list(range(1,len(labels1)+1)))}



# checking how dict looks like 

out1 = dict(list(city_codes.items())[0: 3])

print("dict looks like: " + str(out1))



# for country column..



labels2 = dataset['Country'].astype('category').cat.categories.tolist()

country_codes = {k: v for k,v in zip(labels2,list(range(1,len(labels2)+1)))}



# checking how dict looks like 

out2 = dict(list(country_codes .items())[0: 3])

print("dict looks like: " + str(out2))
# creating country city dataframe to sort further..

country_filter = pd.DataFrame({'Country':dataset.Country, 'Cities': dataset.City})
# creating dict of country + city..which includes Country as key and Cities in that country as values

#.....ignore city names not matching to the real city names in the world..since it is practice purpose dataset..not real data



country_city_dict = country_filter.groupby(['Country'])['Cities'].apply(lambda grp: list(grp.value_counts().index)).to_dict()
import pickle



# Pickling fianl_predict 

filename = 'fianl_model.p'    # provide file name

outfile = open(filename,'wb') #creating empty file..it will saved as model_predict.py in same working directory

pickle.dump(final_model,outfile) # dumping pickled object in file created.  

outfile.close()





# Pickling unique value dictionary..

dict1 = open('country_codes1.p', 'wb') 

pickle.dump(country_codes, dict1)                      

dict1.close() 



dict2 = open('city_codes1.p', 'wb') 

pickle.dump(city_codes, dict2)                      

dict2.close() 



dict3 = open('country_city_dict.p', 'wb')

pickle.dump(country_city_dict, dict3)