# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split



from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn import svm



from sklearn.metrics import precision_recall_curve

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
def plot_pricision_recall(model,X_test,y_test,title_name):

    # -------- get probability of Prediction for true result only using predict_proba func. --------

    y_pred = model.predict_proba(X_test)[:,1]



    # -------- get precision,recall and thresholds to Drawing the curve to clarify result ----------

    precision, recall, thresholds = precision_recall_curve(y_test, y_pred)

    

    # --------------------- Intialize Graph ---------------------

    plt.xlabel("Threshhold")

    plt.ylabel("Precision/Recall")

    plt.title('Precision/Recall '+title_name)



    # --------------------- Drawing the graph ---------------------

    plt.plot(thresholds,precision[:-1],label='Precision')

    plt.plot( thresholds,recall[:-1], label='Recall')



    # --------------------- Set Legend on the graph ---------------------

    plt.legend(bbox_to_anchor=(1.28,1), loc='best', borderaxespad=0.)
# --------------------- Load Dataset using panda ---------------------

data = pd.read_csv("/kaggle/input/autism-screening-for-toddlers/Toddler Autism dataset July 2018.csv")



# --------------------- Remove missing values in panda using dropna func. ---------------------

data = data.dropna()



# --------------------- Adjust columns names for dataset ---------------------

data.columns = [

    "Case_No",

    "A1",

    "A2",

    "A3",

    "A4",

    "A5",

    "A6",

    "A7",

    "A8",

    "A9",

    "A10",

    "Age_Mons",

    "Qchat-10-Score",

    "Sex",

    "Ethnicity",

    "Jaundice",

    "Family_mem_with_ASD",

    "Who_completed_the_test",

    "Class/ASD_Traits",

]



# --------------------- Print first 10 examples in dataset to explore it ---------------------

data.head(10)
# --------------------- Filtering dataset fron case number column ---------------------

data = data.drop(columns=['Case_No','Qchat-10-Score'])
# --------------------- Print out the unique values for each column ---------------------

for column_name in data.columns:

    print(

        """

    {column_name}:

    {unique_values}""".format(

            column_name=column_name,

            unique_values=", ".join(

                map(str, data[column_name].unique())

            ),

        )

    )

    

print(

    """

NUMBER OF EXAMPLES:{}

NUMBER OF COLUMNS: {}

""".format(

        data.shape[0], data.shape[1]

    )

)
# - Draw histogram for each column with class/asd to see which columns Influential in class/ASD -

bin_data=pd.DataFrame.copy(data)



# --------------------- Encode Class/ASD_Traits unique values to 1 or 0 ---------------------

bin_data['Class/ASD_Traits']=bin_data['Class/ASD_Traits'].apply(lambda x:1 if x=="Yes" else 0)



for column_name in data.columns:

    if(column_name == 'Class/ASD_Traits'):

        continue;

    group_by_modelLine = bin_data[[column_name,'Class/ASD_Traits']].groupby(by=[column_name])

# --------------------- get mean to draw histogram ---------------------    

    mean=group_by_modelLine.mean().reset_index()

    mean.columns=[column_name,'positive']

    mean['negative']=mean['positive'].apply(lambda x: 1-x)

    mean=pd.melt(mean, id_vars=column_name, var_name="Pos/Neg", value_name="Class/ASD_Traits")

    p=sns.barplot(x=column_name, y='Class/ASD_Traits', hue='Pos/Neg', data=mean.reset_index())

    p.legend(loc='best', bbox_to_anchor=(1.28, 1), ncol=1)

    plt.show()
# --------------------- The columns will not change ---------------------

org_data = data[

    [

        "A1",

        "A2",

        "A3",

        "A4",

        "A5",

        "A6",

        "A7",

        "A8",

        "A9",

        "A10",

        "Age_Mons",

    ]

]



# --------------------- The columns wich will be encoded using label encoding -------------------

label_data = data[

    [

        "Sex",

        "Jaundice",

        "Family_mem_with_ASD",

        "Class/ASD_Traits",

    ]

]

label_data["Sex"] = label_data["Sex"].apply(

    lambda x: 1 if x == "m" else 0

)

label_data["Jaundice"] = label_data["Jaundice"].apply(

    lambda x: 1 if x == "yes" else 0

)

label_data["Family_mem_with_ASD"] = label_data[

    "Family_mem_with_ASD"

].apply(lambda x: 1 if x == "yes" else 0)

label_data["Class/ASD_Traits"] = label_data[

    "Class/ASD_Traits"

].apply(lambda x: 1 if x == "Yes" else 0)



# --------------------- The columns wich will be encoded using one hot encoding -----------------

one_hot_encoded_data = data[

    ["Ethnicity", "Who_completed_the_test"]

]

one_hot_encoded_data = pd.get_dummies(one_hot_encoded_data)

# --------------------- concatenate all columns ---------------------

final_data = pd.concat([org_data,label_data,one_hot_encoded_data],axis = 1)



# --------------------- Droping column Class/ASD_Traits from features ---------------------

X = final_data.drop(columns=['Class/ASD_Traits'])

y = final_data[['Class/ASD_Traits']]
# --------------------- Splitting data for test data and train data ---------------------

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

X_train, X_val, y_train, y_val   = train_test_split(X_train, y_train, test_size=0.2, random_state=1)
# --------------------- Create LogisticRegression model and train it ---------------------

clf = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial').fit(X_train, y_train)

plt.subplots(1,2,figsize=(15,5))

plt.subplots_adjust(wspace=0.5)

plt.subplot(121)

plot_pricision_recall(clf,X_train,y_train,'LogisticRegression-Train')

plt.subplot(122)

plot_pricision_recall(clf,X_val,y_val,'LogisticRegression-Validation')

plt.show()
# --------------------- RandomForestClassifier ---------------------

rfc= RandomForestClassifier(n_estimators=500)

rfc.fit(X_train,y_train)



plt.subplots(1,2,figsize=(15,5))

plt.subplots_adjust(wspace=0.5)

plt.subplot(121)

plot_pricision_recall(rfc,X_train,y_train,'RandomForest-Train')

plt.subplot(122)

plot_pricision_recall(rfc,X_val,y_val,'RandomForest-Validation')

plt.show()
knn= KNeighborsClassifier(n_neighbors=13)

knn.fit(X_train,y_train)



plt.subplots(1,2,figsize=(15,5))

plt.subplots_adjust(wspace=0.5)

plt.subplot(121)

plot_pricision_recall(knn,X_train,y_train,'KNearestNeighbours-Train')

plt.subplot(122)

plot_pricision_recall(knn,X_val,y_val,'KNearestNeighbours-Validation')

plt.show()
svm = svm.SVC(probability=True)

svm.fit(X_train, y_train) 



plt.subplots(1,2,figsize=(15,5))

plt.subplots_adjust(wspace=0.5)

plt.subplot(121)

plot_pricision_recall(svm,X_train,y_train,'SVM-Train')

plt.subplot(122)

plot_pricision_recall(svm,X_val,y_val,'SVM-Validation')

plt.show()
plt.subplots(1,2,figsize=(15,5))

plt.subplots_adjust(wspace=0.5)

plt.subplot(121)

plot_pricision_recall(clf,X_test,y_test,'LogisticRegression-Test')

plt.subplot(122)

plot_pricision_recall(rfc,X_test,y_test,'RFC-Test')

plt.show()



plt.subplots(1,2,figsize=(15,5))

plt.subplots_adjust(wspace=0.5)

plt.subplot(121)

plot_pricision_recall(knn,X_test,y_test,'KNN-Test')

plt.subplot(122)

plot_pricision_recall(svm,X_test,y_test,'SVM-Test')

plt.show()