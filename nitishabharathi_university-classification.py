import numpy as np 

import pandas as pd 

from collections import defaultdict

from sklearn.model_selection import train_test_split

from lightgbm import LGBMClassifier

from xgboost import XGBClassifier

from sklearn import neighbors, svm,metrics

from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv('../input/university-recommendation/original_data.csv') #original data file

score_table= pd.read_csv('../input/university-recommendation/score.csv') # GRE score conversion table



#Dropping unwanted columns

data = data.drop(['gmatA','gmatQ','gmatV','specialization','department','program','toeflEssay','userProfileLink','topperCgpa','termAndYear','userName','toeflScore','industryExp','internExp','confPubs','journalPubs','researchExp'],1)

# Dropping missing values

data = data.dropna()



#Only admitted data is required, dropping not admitted

data = data[data["admit"] > 0]

data = data.drop("admit", 1)



# Dropping universites whose instances are lesser in number

university_list = list(set(data["univName"].tolist()))

for i in range(len(university_list)):

    if len(data[data["univName"] == university_list[i]]) < 100:

        data = data[data["univName"] != university_list[i]]

def normalize_gpa(data, cgpa, totalcgpa):

    '''

    Utility function to normalize CGPA

    '''

    cgpa = data[cgpa].tolist()

    totalcgpa = data[totalcgpa].tolist()

    for i in range(len(cgpa)):

        if totalcgpa[i] != 0:

            cgpa[i] = cgpa[i] / totalcgpa[i]

        else:

            cgpa[i] = 0

    data["cgpa"] = cgpa

    return data
def feature_extraction_categorical_variable1(data, feature):

    '''

    Utility function to preprocess categorical features

    '''

    feature_list = list(data[feature].astype(str))

    student_id_for_feature = defaultdict(list)

    for i in range(len(feature_list)):

        feature_list[i] = str(feature_list[i])

        feature_list[i] = feature_list[i].strip()

        feature_list[i] = feature_list[i].replace("-", "")

        feature_list[i] = feature_list[i].replace(".", "")

        feature_list[i] = feature_list[i].partition("/")[0]

        feature_list[i] = feature_list[i].partition("(")[0]

        feature_list[i] = feature_list[i].replace(" ", "")

        feature_list[i] = feature_list[i].lower()

    data[feature] = feature_list

    return data
def scoreConversion(feature):

    '''

    Utility function: Gre Old Score to New Score

    '''

    gre_score = list(data[feature])

    for i in range(len(gre_score)):

        if gre_score[i] > 170:

            try:

                if feature =='greV':

                    gre_score[i]=score_table['newV'][gre_score[i]]

                elif feature == 'greQ':

                    gre_score[i]=score_table['newQ'][gre_score[i]]

            except:

                continue

    return gre_score
# Preprocessing each column



data = feature_extraction_categorical_variable1(data, "ugCollege")

data['ugCollege'] = data['ugCollege'].astype('category')

data['ugCollege_code'] = data['ugCollege'].cat.codes



data = feature_extraction_categorical_variable1(data, "major")

data['major'] = data['major'].astype('category')

data['major_code'] = data['major'].cat.codes

data = data.drop(['major','ugCollege'],1)



data = normalize_gpa(data, "cgpa", "cgpaScale")



data['greV'] = data['greV'].astype('int')

data['greQ'] = data['greQ'].astype('int')

score_table.set_index(['old'],inplace=True)

data['greV']=scoreConversion('greV')

data['greQ']=scoreConversion('greQ')

data = data[data['greV']<=170]

data = data[data['greQ']<=170]
x = data.drop(['univName'], 1)

y = data['univName']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)





# Random Forest Classifier



clf=RandomForestClassifier(n_estimators=1000)

clf.fit(x_train,y_train)

y_pred=clf.predict(x_test)





# Support Vector Classifier

clf = svm.SVC()

clf.fit(x_train,y_train)

y_pred=clf.predict(x_test)





# K Nearest Neighbours

clf = neighbors.KNeighborsClassifier(300, weights='uniform')

clf.fit(x_train,y_train)

y_pred=clf.predict(x_test)





# XGBoost Classifier



clf = XGBClassifier()

clf.fit(x_train,y_train)

y_pred=clf.predict(x_test)





# Light GBM Classifier

clf = LGBMClassifier()

clf.fit(x_train,y_train)

y_pred=clf.predict(x_test)
