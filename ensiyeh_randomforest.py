# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
##ketabkhanehaye lazem ra dar in ghesmat import mikonim

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn as sk
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
import Orange3
from sklearn.svm import SVR
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import SVC
##ba'zi az in libraryha dar filehaye digar mannd SVM & SVR & ... estefade shod. k ma behtarin khoruji k b dast avardim upload kardim.


####Orange is a library for preprocessing
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
##dar inja trainset & testset ra az directory marbute mikhanim
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


##dar in ghesmat ghasd dashtim k ba estefade az library orange ruye dataset preprocessing anjam bdahim k code error dad. chon maghadire vijegiha
##peyvaste bud , ghasd dashtim discretization anjam bdahim k b natije naresidim.yani discretization anjam shod vali b ellate missing values
##edame barname momken nashod va ba method marbute ham natavanestim dar inj in moshkel ra raf konim.
#data = Orange.data.Table("../input/train.csv")

#disc_data = Orange.data.discretization.DiscretizeTable(data,method=Orange.feature.discretization.EqualFreq(n=3))

#disc_data = Orange.data.discretization.DiscretizeTable(data,feature=None,method=Orange.feature.discretization.EqualFreq(n=3))

#X = disc_data.values[:,2:20]
#y = disc_data.values[:,21]

#y = y.astype('int')

#imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
#imputer = imputer.fit(X)
#X = imputer.transform(X)


#Xt = test.values[:,2:20]

##khastim pca ra ejra konim k barname error dad.
#pca = PCA(n_components=2)
#
#selection = SelectKBest(k=1)
###
#combined_features = FeatureUnion([("pca", pca), ("univ_select", selection)])
#
#X_features = combined_features.fit(X, y).transform(X)

##dar in ghesmat recordhaye vijegi nemuneha dar trainset ra dar X gharar midahim va labelhaye anha ra dar y gharar midahim.
X = train.values[:,2:20]
y = train.values[:,21]
y = y.astype('int')

#######preprocessing on trainset...dar inja valuehaii k miss shodan ra meghdardehi mikonim. ba strategy miangin giri.
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X)
X = imputer.transform(X)


##vijegihaye testset ra dar Xt gharar midahim. 
Xt = test.values[:,2:20]

#####preprocessing on testset...dar inja valuehaii k dar testset miss shodan ra meghdardehi mikonim. moshabehe trainset
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(Xt)
Xt = imputer.transform(Xt)

##########another preprocessing...in ghesmat baraye estandard saziye valueha b kar miravad
sc = StandardScaler()
sct = StandardScaler()
X = sc.fit_transform(X)
Xt = sc.transform(Xt)


##dar inja model morede nazar ramisazim. ma classifierhaye mokhtalef ra test kardim va dar nahayat behtarin deghat ra ruye testset baraye randomforest
##k andazeye an 200 bud b dast avardim. bishtar ya kamtar az in tedad deghat ra kahesh midad.
clf = RandomForestClassifier(n_estimators = 200 , criterion = 'entropy')
##modele sakhte shode ra ruye trainset fit mikonim
clf.fit(X, y)



##dar in ghesmat ba estefade az modele b dast amade label nemunehaye testset ra predict mikonim.
cols = { 'PlayerID': [i+901 for i in range(440)] , 'TARGET_5Yrs': [clf.predict([Xt[i]])[0] for i in range(440)] }


##natayej ra dar submission gharar midahim
submission = pd.DataFrame(cols)


##nemayesh natayej
print(submission)

submission.to_csv("submission.csv", index=False)

