!pip install pyforest
from pyforest import *
df= pd.read_csv("../input/districtwise-ground-water-resources-by-july-2017/Dynamic_2017_2_0.csv")
df.head()
df=df.drop("S.no.", axis=1)
df.head()
df.isnull().sum()
df= df.dropna(axis=0)
df.dtypes
df["net available water present"]= df["Total Annual Ground Water Recharge"] - df["Total Current Annual Ground Water Extraction"]
df.head()
ax=sns.barplot(x="net available water present", y="Name of State", data=df)

ax.set_yticklabels(ax.get_yticklabels(), rotation=0, horizontalalignment='right')


!pip install https://github.com/pandas-profiling/pandas-profiling/archive/master.zip
from pandas_profiling import ProfileReport
profile = ProfileReport(df, title='District wise ground water Stats', html={'style':{'full_width':False}})
profile