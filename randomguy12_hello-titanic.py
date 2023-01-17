

import numpy as np 

import pandas as pd 

from sklearn.ensemble import RandomForestClassifier

from patsy import dmatrices

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.




df = pd.read_csv("../input/train.csv")



df.drop(["Ticket", "Cabin"], axis=1)

df.dropna()

# Create an acceptable formula for our machine learning algorithms

formula_ml = 'Survived ~ C(Pclass) + C(Sex) + Age + SibSp + Parch + C(Embarked)'

    

y,x = dmatrices(formula_ml, data=df, return_type='dataframe')  

y = np.asarray(y).ravel()

forest = RandomForestClassifier()

forest.fit(x, y)

forest.score(x, y)