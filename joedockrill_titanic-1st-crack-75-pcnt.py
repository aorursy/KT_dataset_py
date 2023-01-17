import numpy as np

import pandas as pd

import statsmodels.api as sm

import matplotlib.pyplot as plt



import seaborn as sns

sns.set()



import os

import re

import statistics

import joe_helpers as joe
def ExtractTitle(name):

###############################################################################################

# extracts a title from a name field, eg: "Braund, Mr. Owen Harris" returns "Mr"

###############################################################################################

    ptn = "[^,]*,\s(the\s)?([^\.]*)"

    p = re.compile(ptn, re.IGNORECASE)

    m = p.match(name)



    if m is not None:

        return m.group(2)

    else:

        print("FAILED EXTRACTING TITLE FROM: " + name)

        return None



def ProcessTitles(cln):

###############################################################################################

# sets the title col on all rows (and prints out all the titles)

###############################################################################################

    titles = {}

    t = ""

    count = 0



    for index, row in cln.data.iterrows():

        t = ExtractTitle(row.Name)



        if t is not None:

            cln.data.loc[index, "Title"] = t

            

            if t in titles:

                titles[t] += 1

            else:

                titles[t] = 1



    #print(titles)

 

def ProcessNobles(cln):

###############################################################################################

# sets the Noble flag on all rows with nobility titles, this may not be useful

###############################################################################################

    nobles = {

        'Don': 0, 

        'Dona': 0, 

        'Major': 0, 

        'Lady': 0, 

        'Sir': 0, 

        'Col': 0, 

        'Countess': 0, 

        'Jonkheer': 0

    }

    

    for index, row in cln.data.iterrows():

        if row.Title in nobles:

            cln.data.loc[index, "Noble"] = 1    



###############################################################################################

# all titles found in the data, split male/female -- aquired from ProcessTitles()

# need to check these against test and make sure we don't see any new ones

# at the moment we're only using these for ProcessNobles()

###############################################################################################

all_titles = ['Mr', 'Mrs', 'Miss', 'Master', 'Don', 'Rev', 'Dr', 'Mme', 'Ms', 'Major', 'Lady', 'Sir', 'Mlle', 'Col', 'Capt', 'Countess', 'Jonkheer', 'Dona']

m_titles = ['Mr', 'Master', 'Don', 'Rev', 'Dr', 'Major', 'Sir', 'Col', 'Capt', 'Jonkheer']

f_titles = ['Mrs', 'Miss', 'Mme', 'Ms', 'Lady', 'Mlle', 'Countess', 'Dona']



def ProcessSexPlus(cln):

###############################################################################################

# add SexPlus col (expands male/female to man/woman/boy/girl)

###############################################################################################

    df = cln.data

    

    df.loc[(df.Sex == "male") & (df.Age < 16), ["SexPlus"]] = "boy"

    df.loc[(df.Sex == "male") & (df.Age >= 16), ["SexPlus"]] = "man"

    

    df.loc[(df.Age.isnull()) & (df.Title == "Master"), ["SexPlus"]] = "boy"

    df.loc[(df.SexPlus == "") & (df.Sex == "male"), ["SexPlus"]] = "man"

    

    df.loc[(df.Sex == "female") & (df.Age < 16), ["SexPlus"]] = "girl"

    df.loc[(df.Sex == "female") & (df.Age >= 16), ["SexPlus"]] = "woman"

    

    #Miss and Mlle could be handled better, we could look at the percentage of Miss and Mlle with ages set

    #who are actually child vs unmarried adult, then split the nulls between girl and woman

    df.loc[(df.Age.isnull()) & (df.Title == "Miss"), ["SexPlus"]] = "girl"

    df.loc[(df.Age.isnull()) & (df.Title == "Mlle"), ["SexPlus"]] = "girl"

    df.loc[(df.SexPlus == "") & (df.Sex == "female"), ["SexPlus"]] = "woman"



def ProcessFamilies(cln):

###############################################################################################

# process family groups, split SibSp/Parch out to Spouse/Children/Parents/Siblings

###############################################################################################

    df = cln.data

    df_process = df.loc[(df.SibSp > 0) | (df.Parch > 0)]

    

    for index, row in df_process.iterrows():

        process_as = ""

        

        if row.SibSp > 1 or row.SexPlus == "boy" or row.SexPlus == "girl":

            process_as = "child"

        elif row.Parch > 2 or row.SexPlus == "man" or row.SexPlus == "woman":

            process_as = "adult"

            

        #handful of annoying edge cases with < 16 years old married girls...

        if row.SibSp == 1 and row.Parch == 0 and row.Title == "Mrs" and row.SexPlus == "girl":

            process_as = "adult"



        if process_as == "child":

            df.loc[index, "Parents"] = row.Parch

            df.loc[index, "Siblings"] = row.SibSp

            

            if row.SexPlus == "":

                #if we don't have an age so haven't set a SexPlus we can do it here

                if row.Sex == "male":

                    df.loc[index, "SexPlus"] = "boy"

                else:

                    df.loc[index, "SexPlus"] = "girl"

        elif process_as == "adult":

            df.loc[index, "Spouse"] = row.SibSp

            df.loc[index, "Children"] = row.Parch

            

            if row.Sex == "male":

                df.loc[index, "SexPlus"] = "man"

            else:

                df.loc[index, "SexPlus"] = "woman"



        df.loc[index, "FamilySize"] = (1 + row.Parch + row.SibSp)      

    

def ProcessMissingAges(cln):

###############################################################################################

# process the rows with missing ages using medians of the 80% which do have Age set

# -- must be called after ProcessTitles() and ProcessSexPlus()

###############################################################################################

    df = cln.data



    #originally i used median ages for man/woman/boy/girl for the missing age values but after plotting it

    #became clear that these were different across passenger classes so now i use the median from each class



    for i in range(1,4):

        median_man   = statistics.median(df.loc[(df.Pclass == i) & (df.Age.isnull() == False) & (df.SexPlus == "man") , "Age"])

        median_boy   = statistics.median(df.loc[(df.Pclass == i) & (df.Age.isnull() == False) & (df.SexPlus == "boy") , "Age"])

        median_woman = statistics.median(df.loc[(df.Pclass == i) & (df.Age.isnull() == False) & (df.SexPlus == "woman"), "Age"])

        median_girl  = statistics.median(df.loc[(df.Pclass == i) & (df.Age.isnull() == False) & (df.SexPlus == "girl") , "Age"])



        df.loc[(df.Pclass == i) & (df.Age.isnull()) & (df.SexPlus == "man"), ["Age"]] = median_man

        df.loc[(df.Pclass == i) & (df.Age.isnull()) & (df.SexPlus == "woman"), ["Age"]] = median_woman

        df.loc[(df.Pclass == i) & (df.Age.isnull()) & (df.SexPlus == "boy"), ["Age"]] = median_boy

        df.loc[(df.Pclass == i) & (df.Age.isnull()) & (df.SexPlus == "girl"), ["Age"]] = median_girl

        

def ProcessFarePerPerson(cln):

###############################################################################################

# calculate fare per person based on the number of people on the ticket

###############################################################################################

    cln.SetMissingValues("Fare", 0)

    df = cln.data

    tickets_count = df.groupby("Ticket")["Ticket"].count()

            

    for ticket_number in tickets_count.keys():

        fpp = df.loc[df.Ticket == ticket_number, "Fare"] / tickets_count[ticket_number]

        df.loc[df.Ticket == ticket_number, "FarePerPerson"] = fpp

        

def ProcessBins(cln):

###############################################################################################

# bin age and fare

###############################################################################################

    cln.data["FareBin"] = pd.qcut(cln.data.FarePerPerson, 4, labels=[1, 2, 3, 4])

    cln.data["AgeBin"] = pd.qcut(cln.data.Age, 5, labels=[1, 2, 3, 4, 5])

    cln.data.AgeBin = cln.data.AgeBin.astype("int64")

    cln.data.FareBin = cln.data.FareBin.astype("int64")
###############################################################################################

# run preprocessing

###############################################################################################



def Clean(file1, file2):

###############################################################################################

# load the data and add our new cols 

###############################################################################################

    cln = joe.DataCleaner(file1)

    cln.Concat(file2)    



    cln.AddColumn("Title", "")

    cln.AddColumn("Noble", 0)

    cln.AddColumn("SexPlus", "")

    cln.AddColumn("Spouse", 0)

    cln.AddColumn("Children", 0)

    cln.AddColumn("Parents", 0)

    cln.AddColumn("Siblings", 0)

    cln.AddColumn("FamilySize", 0)

    cln.AddColumn("FarePerPerson", 0)



    ProcessTitles(cln)

    ProcessNobles(cln)

    ProcessSexPlus(cln)

    ProcessFamilies(cln)

    ProcessMissingAges(cln)

    ProcessFarePerPerson(cln)

    ProcessBins(cln)

    

    return cln.data



df_all = Clean("/kaggle/input/titanic/train.csv", "/kaggle/input/titanic/test.csv")



###############################################################################################

# set up dataframe slices for some plotting

###############################################################################################

df_train = df_all.loc[df_all.Survived.isnull() == False]

df_test = df_all.loc[df_all.Survived.isnull() == True]



df_male = df_train.loc[df_train.Sex == "male"]

df_female = df_train.loc[df_train.Sex == "female"]



df_men = df_train.loc[df_train.SexPlus == "man"]

df_women = df_train.loc[df_train.SexPlus == "woman"]

df_children = df_train.loc[(df_train.SexPlus == "boy") | (df_train.SexPlus == "girl")]

df_boys = df_train.loc[df_train.SexPlus == "boy"]

df_girls = df_train.loc[df_train.SexPlus == "girl"]



df_1st = df_train.loc[df_train.Pclass == 1]

df_2nd = df_train.loc[df_train.Pclass == 2]

df_3rd = df_train.loc[df_train.Pclass == 3]



df_s = df_train.loc[df_train.Embarked == "S"]

df_c = df_train.loc[df_train.Embarked == "C"]

df_q = df_train.loc[df_train.Embarked == "Q"]
def FmtPcnt(num, den, rnd=1):

###############################################################################################

# format a percentage string

###############################################################################################

        return str(round((num / den * 100), rnd)) + "%"



def DrawDSHist(df, title, **kwargs):

###############################################################################################

# draw a single died/survived histogram

###############################################################################################

    ntotal = len(df)

    ndied = len(df.loc[df.Survived == 0])

    nsurvived = ntotal - ndied

    sdied = "Died (" + FmtPcnt(ndied, ntotal) + ")"

    ssurvived = "Survived (" + FmtPcnt(nsurvived, ntotal) + ")"

    

    t = title + " (" + str(len(df)) + " total)"

    joe.Plotting(df).CountPlot("Survived", t, tick_locations=[0,1], tick_labels=[sdied, ssurvived], **kwargs)



def DrawDSFacetGrid(df, col, title):

###############################################################################################

# draw a died/survived facetgrid

###############################################################################################

    joe.Plotting(df).FacetGrid(None, "Survived", col, sns.distplot, title=title, kde=False, bins=[0,1,2])



def DrawAgeRegFacetGrid(df, col, title):

###############################################################################################

# draw a died/survived facetgrid

###############################################################################################

    joe.Plotting(df).FacetGrid("Age", "Survived", col, sns.regplot, title=title, logistic=True, ci=None, line_kws={'color':'orange'})

    

def DrawPredictedSurvival(df, col, title):

###############################################################################################

# draw a logistic regression curve to predict survival by col

###############################################################################################

    joe.Plotting(df).RegressionPlot(col, "Survived", title=title, logistic=True, ci=None)
#temp hack until i fix my plotting wrapper

os.mkdir("/kaggle/working/plots")
DrawDSHist(df_train, "All")

DrawDSHist(df_train, "All by Sex", hue="Sex")

DrawDSHist(df_train, "All by Class", hue="Pclass")

DrawDSHist(df_train, "All by SexPlus", hue="SexPlus", hue_order=["man", "woman", "boy", "girl"])

DrawDSHist(df_1st, "1st class by SexPlus", hue="SexPlus", hue_order=["man", "woman", "boy", "girl"])

DrawDSHist(df_2nd, "2nd class by SexPlus", hue="SexPlus", hue_order=["man", "woman", "boy", "girl"])

DrawDSHist(df_3rd, "3rd class by SexPlus", hue="SexPlus", hue_order=["man", "woman", "boy", "girl"])
joe.Plotting(df_train).BoxPlot(x="Pclass", y="Age", title="All by Pclass")

#joe.Plotting(df_1st).BoxPlot(x="SexPlus", y="Age", order=["man","woman","boy","girl"], title="1st by SexPlus")

#joe.Plotting(df_2nd).BoxPlot(x="SexPlus", y="Age", order=["man","woman","boy","girl"], title="2nd by SexPlus")

#joe.Plotting(df_3rd).BoxPlot(x="SexPlus", y="Age", order=["man","woman","boy","girl"], title="3rd by SexPlus")
DrawPredictedSurvival(df_men, "Age", "Men")

DrawPredictedSurvival(df_women, "Age", "Women")

DrawPredictedSurvival(df_boys, "Age", "Boys")

DrawPredictedSurvival(df_girls, "Age", "Girls")



DrawPredictedSurvival(df_1st, "Age", "1st Class")

DrawPredictedSurvival(df_2nd, "Age", "2nd Class")

DrawPredictedSurvival(df_3rd, "Age", "3rd Class")
DrawDSHist(df_train, "All by AgeBin", hue="AgeBin")

DrawDSHist(df_train, "All by FareBin", hue="FareBin")
#S = Southampton

#C = Cherbourg (France)

#Q = Queenstown (Ireland)



DrawDSHist(df_train, "All by Embarked", hue="Embarked")

DrawDSHist(df_train, "All by Pclass", hue="Pclass")

DrawDSHist(df_s, "Southampton by Pclass", hue="Pclass")

DrawDSHist(df_c, "Cherbourg by Pclass", hue="Pclass")

DrawDSHist(df_q, "Queenstown by Pclass", hue="Pclass")



df.groupby(["Embarked", "Pclass"])["Embarked"].count()
DrawPredictedSurvival(df_men, "FamilySize", "Family Size (Men)")

DrawPredictedSurvival(df_women, "FamilySize", "Family Size (Women)")

DrawPredictedSurvival(df_boys, "FamilySize", "Family Size (Boys)")

DrawPredictedSurvival(df_girls, "FamilySize", "Family Size (Girls)")
from sklearn.model_selection import KFold, cross_val_score, train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import PolynomialFeatures

from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.compose import ColumnTransformer



def CreateModel(x):

###############################################################################################

# create the model pipeline

###############################################################################################

    num_features = x.select_dtypes(include='number').columns.to_list()

    cat_features = x.select_dtypes(include='object').columns.to_list()



    num_pipe = Pipeline([

        ('imputer', SimpleImputer(strategy='median')),

        ('poly',PolynomialFeatures(degree=2)),

        ('scaler', StandardScaler())

    ])

    

    cat_pipe = Pipeline([

        ('imputer', SimpleImputer(strategy='most_frequent')),

        ('coder', OneHotEncoder(handle_unknown='ignore'))

    ])



    ctr = ColumnTransformer(remainder='drop',

        transformers=[

        ('numerical', num_pipe, num_features),

        ('categorical', cat_pipe, cat_features)

    ])



    model = Pipeline([

        ('transformer', ctr),

        ('predictor', RandomForestClassifier(n_jobs=1,random_state=0))

    ])

    

    return model



def TestModel(data):

###############################################################################################

# test the model with a train/test split and a k-fold cv

###############################################################################################

    md = joe.DataCleaner(data.copy())

    y = md.data["Survived"]

    md.DropColumns("Survived")

    x = md.data



    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=456)

    

    model = CreateModel(x_train)

    model.fit(x_train, y_train)

    

    #score...

    print("------------------------------------")

    print("Default score: ", model.score(x_train, y_train))

    kfold = KFold(n_splits=10, shuffle=True)

    kf_cv_scores = cross_val_score(model, x_train, y_train, cv=kfold)

    print("KFold average: %.2f" % kf_cv_scores.mean())

    print("------------------------------------")

    print("")

    

def GetPredictions(df_train, df_test):

###############################################################################################

# run the model and get the predictions

###############################################################################################

    trn = joe.DataCleaner(df_train.copy())

    y = trn.data["Survived"]

    trn.DropColumns("Survived")

    x = trn.data

    

    model = CreateModel(x)

    model.fit(x, y)



    tst = joe.DataCleaner(df_test.copy())

    tst.DropColumns("Survived")

    x = tst.data

    

    pred_values = model.predict(x)

    

    tst.data["PassengerId"] = df_test.PassengerId

    tst.data["Survived"] = pred_values

    tst.data.Survived = tst.data.Survived.apply(round)

    

    return tst.data



def ModifyDF(df):

###############################################################################################

# modify a df for modelling

###############################################################################################

    tst = joe.DataCleaner(df.copy())    

    tst.DropColumns(["Name", "Age", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked", "Title", "Noble", "Spouse", "Children", "Parents", "Siblings"])    

    

    return tst.data
###############################################################################################

# split class model - test / create submission

# i thought doing each Pclass seperately might be better but it didn't work :(

###############################################################################################

mod_train_1st = ModifyDF(df_1st)

mod_train_2nd = ModifyDF(df_2nd)

mod_train_3rd = ModifyDF(df_3rd)



mod_test_1st = ModifyDF(df_test.loc[df_test.Pclass == 1])

mod_test_2nd = ModifyDF(df_test.loc[df_test.Pclass == 2])

mod_test_3rd = ModifyDF(df_test.loc[df_test.Pclass == 3])



print("1st class")

TestModel(mod_train_1st) 

print("2nd class")

TestModel(mod_train_2nd) 

print("3rd class")

TestModel(mod_train_3rd)



#sub_1st = GetPredictions(mod_train_1st, mod_test_1st, mod_cols_1st)

#sub_2nd = GetPredictions(mod_train_2nd, mod_test_2nd, mod_cols_2nd)

#sub_3rd = GetPredictions(mod_train_3rd, mod_test_3rd, mod_cols_3rd)



#df_submission = pd.concat([sub_1st, sub_2nd, sub_3rd], ignore_index=True)

#df_submission.reset_index(drop=True)

#df_submission = pd.DataFrame(df_submission, columns=["PassengerId", "Survived"])

#df_submission.to_csv("submission.csv", index=False)
###############################################################################################

# single model - test / create submission

###############################################################################################

mod_train = ModifyDF(df_train)

mod_test = ModifyDF(df_test)



TestModel(mod_train) 



df_submission = GetPredictions(mod_train, mod_test)

df_submission = pd.DataFrame(df_submission, columns=["PassengerId", "Survived"])

df_submission.to_csv("submission.csv", index=False)