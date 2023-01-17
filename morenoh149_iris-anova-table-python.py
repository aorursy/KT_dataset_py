import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
df = pd.read_csv('../input/Iris.csv')
df.head()
df.describe()
iris_lm_one_categorical=ols('SepalLengthCm ~ C(Species)', data=df).fit() #Specify C for Categorical
sm.stats.anova_lm(iris_lm_one_categorical, typ=2)
iris_lm=ols('SepalLengthCm ~ C(Species) + SepalWidthCm + PetalLengthCm + PetalWidthCm', data=df).fit() #Specify C for Categorical
sm.stats.anova_lm(iris_lm, typ=2)
iris_lm.summary()