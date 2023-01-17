# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#more imports; mine, not there by default

import matplotlib.pyplot as plt



from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import LinearRegression



from sklearn.tree import DecisionTreeRegressor

from sklearn.tree import DecisionTreeClassifier



from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import RandomForestClassifier



from sklearn.model_selection import validation_curve

from sklearn.model_selection import learning_curve



from sklearn.decomposition import PCA



from sklearn.svm import SVC



#from sklearn.neural_network import MLPClassifier

#seems to take a long time to load
df = pd.read_csv("../input/train.csv")

df[:3]
#find columns w nas

print(df.columns[pd.isnull(df).any()].tolist())

#Age, Cabin, Embarked
def fillcabin(df):

    df.Cabin.fillna(value = "Z0", inplace = True)

fillcabin(df)
#function for creating dummy variables

def dummy(df, source_name, out_name, mapping = None, default = 0):

    #dummies out source_name column to out_name column appended to dict

    #using mapping defined by dict

    invec = list(df[source_name])

    outvec = []

    errlist = []

    

    #flag for default behavior when not given dict - assign values to integers as they are encountered

    given_map = True

    if(mapping == None):

        mapping = {}

        given_map = False

        next_int = 0

    

    

    for val in invec:

        try:

            to_add = mapping[val]

        except:

            if given_map:

                #if value unexpectedly not covered by dict, map to default value and print message

                to_add = default

                vstr = str(val)

                if vstr not in errlist:

                    errlist.append(vstr)

            else:

                mapping[val] = next_int

                to_add = next_int

                next_int += 1

                

        outvec.append(to_add)

    if(len(errlist) > 0):     

        print("Note: these unexpected values were encountered and mapped to 0: " + str(errlist))

        

    df[out_name] = outvec
#dummy sex

dummy(df, 'Sex', 'SexInd')





#dummy embarkment into 'cherbourg or not' and 'queenstown or not'

#default, Southampton, has vast majority of embarkments

#so it makes sense to map nans to it as well

cherb_dict = {'C':1, 'Q':0,'S':0}

qtown_dict = {'C':0, 'Q':1,'S':0}



dummy(df, 'Embarked', 'EmCherbourg', cherb_dict)

dummy(df, 'Embarked', 'EmQueenstown', qtown_dict)
df[:3]
#create dummy variable for cabin letter (which I believe is deck)

#I believe the deck letters go from top to bottom of ship

#so converting them to a linear variable seems reasonable

#it's less obvious what to do with passengers with no cabin, and the one guy whose cabin is 'T'



#for now, letters a thru g = numbers 1 thru 7, and z,t = 7

#later, should separate 'f's from 'f g's

#and double check that letter ~ deck height

#and see where non-cabin passengers would have been (on deck?)



def makedecks(df):

    cabdeck = []

    approx_cabnum =[]

    cabintrans = {'Z':8, 'T':8, 'A':1, 'B':2, 'C':3, 'D':4, 'E':5, 'F':6, 'G':7}

    for cab in df['Cabin']:

        clet = cab[0]

        cabdeck.append(cabintrans[clet])

           

        #try last 3, then last 2, then last 1 character of cabin variable as cabin num

        try:cnum = int(cab[-3])

        except:

            try:cnum = int(cab[-2])

            except:

                try:cnum = int(cab[-1])

                except:cnum = 0

            

        approx_cabnum.append(cnum)

        

    #fill in unknown cabin numbers with average

    approx_cabnum = np.asarray(approx_cabnum)

    cabavg = int(np.mean(approx_cabnum[np.nonzero(approx_cabnum)]))

    approx_cabnum[approx_cabnum == 0] = round(cabavg)

    

    

    df['CabinDeck'] = cabdeck

    df['CabinNum'] = approx_cabnum

    

makedecks(df)

df[:3]
inputs = ['Pclass', 'Age','SibSp','Parch','Fare', 'SexInd', 'EmCherbourg', 'EmQueenstown', 'CabinDeck', 'CabinNum']

def impute_var(dfin, varname = 'Age', varstouse = inputs):

    df = dfin.copy()

    #need to fill nans of age

    needvar = df.loc[np.isnan(df[varname])]

    #print(needvar)

    #okay, it has 177 nans

    #best thing to do would be to impute ages based on other factors (fare, class, sibsp, parch, cabin, embarked, sexind, cabindeck)



    dfa = df.loc[np.isfinite(df[varname])]



    varinputs = list(varstouse)

    try:

        varinputs.remove(varname)

    except:

        pass





    vreg = DecisionTreeRegressor()

    vreg.fit(dfa[varinputs], dfa[varname])

    print("Regression score on known values: " + str(vreg.score(dfa[varinputs], dfa[varname])))

    

    varimpute = vreg.predict(needvar[varinputs])

    df.loc[np.isnan(df[varname]), varname] = varimpute

    

    return df
df2 = impute_var(df, 'Age')
#normalize df columns, then run PCA

df3 = df2[inputs]

dfn = (df3 - df3.mean())/np.sqrt(df3.var())



pca = PCA()

pca.fit(dfn)

exp = pca.explained_variance_ratio_

cums = np.cumsum(exp)

cums



#we find that, sadly, we can't much reduce the dimensionality; the variance is pretty evenly distributed

#cumulative variance: [ 0.25752306,  0.43920319,  0.54482012,  0.64924261,  0.74149226, 0.81790732,  0.88137508,  0.9338272 ,  0.97655009,  1.        ]





Xbase = np.asarray(df2[inputs])

XT = pca.transform(Xbase)

X = XT



y = df2['Survived']
X = Xbase
#X, y = df2[inputs], df2['Survived']



#for decision trees:

##tovary = "max_depth"

##vary_range = range(1,15)

##valid curve has maxes around 3 and 5-6 (both w scores of slightly above .8)



##try min Gini impurity of split; seems like a good value to vary

##note - seems like results don't differ much with enforcement of max depth 8

#tovary = "min_impurity_split"

#p_right_range = np.arange(.70,.9875,.0125)

#vary_range = 2*p_right_range-2*(p_right_range**2)



#varying this along w max_depth gives a maximum validation score a little below .82

#reached at max_depth ~6 (6,8 give similar results; 3 has a slightly lower maximum)

#for impurity threshold ~ .2, depending a bit on max depth; around .1875 for depth = 6 

#a bit higher for depth = 8, around .2 or .2125 

#max score for depth = 8 is a bit higher, so I'll use this

opt_depth = 8

opt_minsplit = .2125









##for AdaBoost

#tovary = "learning_rate"

#vary_range = np.arange(.1,1.,.1)

#not much difference; will juse use 1 for now



#tovary = "n_estimators"

#vary_range = range(10,400,30)

#best ~ 63; not much variation here either





#for Random Forest

#tovary = "n_estimators"

#vary_range = range(50,250,25)

##not huge variation - best = 100, ~.82





tovary = "min_impurity_split"

##p_right_range = np.arange(.51,.99,.03)

p_right_range = np.arange(.66, .99, .01)

vary_range = 2*p_right_range-2*(p_right_range**2)

##not huge variation, but a big falloff after split value ~ .4; max around .21, value ~.82

##will just keep this at default (~0) value; doesn't seem to cause overtraining



#tovary = "max_depth"

#vary_range = range(3,15,3)

##plateau at 6 thru 9 gets to max value of a little over .82

##num samples is 891; makes sense that allowing over than many leaves leads to overfitting

##use 9 for now



#train_scores, valid_scores = validation_curve(DecisionTreeClassifier(max_depth = 8), X, y, tovary, vary_range, cv=5)

#train_scores, valid_scores = validation_curve(AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),n_estimators = 63), X, y, tovary, vary_range)

train_scores, valid_scores = validation_curve(RandomForestClassifier(n_estimators = 200), X, y, tovary, vary_range)



trainavg = np.mean(train_scores, axis=1)

validavg = np.mean(valid_scores, axis=1)



plt.clf()

plt.plot(vary_range, trainavg, label='train scores')

plt.plot(vary_range, validavg, label='valid scores')

plt.show
#learning curve

train_sizes, train_scores, valid_scores = learning_curve((AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), learning_rate = .75, n_estimators = 400)), X, y)

    #RandomForestClassifier(n_estimators = 100, max_depth = 9), X, y)



trainavg = np.mean(train_scores, axis=1)

validavg = np.mean(valid_scores, axis=1)



plt.clf()

plt.plot(train_sizes, trainavg, label='train scores')

plt.plot(train_sizes, validavg, label='valid scores')

plt.show
#clf = DecisionTreeClassifier(max_depth = opt_depth, min_impurity_split = opt_minsplit)

#clf.fit(X,y)

#clf.score(X,y)
#clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), learning_rate = 1., n_estimators = 63)

#clf.fit(X,y)



#clf.score(X,y)
#clf = RandomForestClassifier(n_estimators = 100, min_impurity_split = .15, max_depth = 20)

clf = RandomForestClassifier(n_estimators = 100, max_depth = 9)



clf.fit(X, y)



clf.score(X, y)
#bring in test data and clean it up



dft = pd.read_csv("../input/test.csv")

dft[:3]
#find columns w/ nans

print(dft.columns[pd.isnull(dft).any()].tolist())

#'age', 'fare', 'cabin'    
fillcabin(dft)



dummy(dft, 'Sex', 'SexInd')

dummy(dft, 'Embarked', 'EmCherbourg', cherb_dict)

dummy(dft, 'Embarked', 'EmQueenstown', qtown_dict)



makedecks(dft)



truncinpts = list(inputs)

truncinpts.remove('Fare')



dft = impute_var(dft,'Age', truncinpts)

dft = impute_var(dft, 'Fare', inputs)
preds = clf.predict(dft[inputs])
#attempt at writing to file w pandas builtin function

pred_dict = {'PassengerId': dft['PassengerId'], 'Survived':preds}

dfp = pd.DataFrame.from_dict(pred_dict)

dfp.to_csv("titanic_preds.csv", index=False)
#attempt at writing to file, taken from Kaggle example code

import csv

prediction_file = open("basic_dtree.csv", "w")

prediction_file_object = csv.writer(prediction_file)



prediction_file_object.writerow(["PassengerId", "Survived"])

for i in range(len(preds)):                                      

    prediction_file_object.writerow([dft['PassengerId'][i],preds[i]]) #write ID and prediction

prediction_file.close()
#print out predictions

print("PassengerId,Survived")

for i in range(len(preds)):                        

    print(str(dft['PassengerId'][i]) + "," + str(preds[i])) #write ID and prediction