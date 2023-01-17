import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
%matplotlib inline
import pandas as pd
import seaborn as sns
sns.set(style="white",color_codes=True)
plt.rcParams['figure.figsize'] = (15,9.27)
# Set the font set of the latex code to computer modern
matplotlib.rcParams['mathtext.fontset'] = "cm"
df = pd.read_csv('../input/train.csv')

titanic = df.drop('Name',axis=1)

titanic.drop(['Ticket','Cabin','Embarked','PassengerId'],axis=1,inplace=True)

def encode(x):
    if x == 'male':
        return 1
    else:
        return 0

titanic['ismale'] = titanic.Sex.apply(encode)
titanic.drop('Sex',axis=1,inplace=True)
titanic.dropna(inplace=True)

titanic.head()
def feature_importance_plot(df,target):
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier()
    y = df[target]
    x = df.drop(target,axis=1)
    model.fit(x,y)
    res_df = pd.DataFrame({'feature':x.columns,'importance':model.feature_importances_})
    res = res_df.sort_values('importance',ascending=False)
    res['cum_importance'] = res.importance.cumsum()
    plt.subplot(211)
    sns.barplot(res.feature,res.importance)
    plt.subplot(212)
    plt.plot(np.arange(1,res.shape[0]+1),res.cum_importance,linewidth=2)
    return(res)
feature_importance_plot(titanic,'Survived')
iris = sns.load_dataset('iris')

feature_importance_plot(iris,'species')
