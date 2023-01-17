import pandas as pd
import seaborn as sns
train = pd.read_csv("../input/titanic/train.csv")
test = pd.read_csv("../input/titanic/test.csv")
#combining train and test
dataset =  pd.concat(objs=[train, test], axis=0,ignore_index=True)
dataset["Fare"].fillna(dataset["Fare"].median(),inplace=True)
#dataset.Fare.isnull().sum()
t=sns.distplot(dataset["Fare"],label="Skewness: %.2f"%(dataset["Fare"].skew()) )
t.legend()
import numpy as np
Log_Fare = dataset["Fare"].map(lambda i: np.log(i) if i > 0 else 0)
t=sns.distplot(Log_Fare,label="Skewness: %.2f"%(Log_Fare.skew()) )
t.legend()
from scipy import stats
Boxcox_Fare = dataset["Fare"].map(lambda i: np.abs(i) if i < 0 else (i+1 if i==0 else i))
Boxcox_Fare= stats.boxcox(Boxcox_Fare)
Boxcox_Fare= pd.Series(Boxcox_Fare[0])
t=sns.distplot(Boxcox_Fare,label="Skewness: %.2f"%(Boxcox_Fare.skew()) )
t.legend()
Sqrt_Fare = dataset["Fare"].map(lambda i: np.sqrt(i))
t=sns.distplot(Sqrt_Fare,label="Skewness: %.2f"%(Sqrt_Fare.skew()) )
t.legend()