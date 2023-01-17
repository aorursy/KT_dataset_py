# Import all required modules
import pandas as pd
import numpy as np

# Disable warnings
import warnings
warnings.filterwarnings("ignore")

# Import plotting modules
import seaborn as sns
sns.set()
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker
%matplotlib inline
# Tune the visual settings for figures in `seaborn`
sns.set_context(
    "notebook", 
    font_scale=1.5,       
    rc={ 
        "figure.figsize": (16, 12), 
        "axes.titlesize": 18 
    }
)

from matplotlib import rcParams
rcParams['figure.figsize'] = 16, 12
df = pd.read_csv('../input/mlbootcamp5_train.csv')
df_base = pd.read_csv('../input/mlbootcamp5_train.csv')
print('Dataset size: ', df.shape)
df.head()
df_uniques = pd.melt(frame=df, value_vars=['gender','cholesterol', 
                                           'gluc', 'smoke', 'alco', 
                                           'active', 'cardio'])
df_uniques = pd.DataFrame(df_uniques.groupby(['variable', 
                                              'value'])['value'].count()) \
    .sort_index(level=[0, 1]) \
    .rename(columns={'value': 'count'}) \
    .reset_index()

sns.factorplot(x='variable', y='count', hue='value', 
               data=df_uniques, kind='bar', size=12);
df_uniques = pd.melt(frame=df, value_vars=['gender','cholesterol', 
                                           'gluc', 'smoke', 'alco', 
                                           'active'], 
                     id_vars=['cardio'])
df_uniques = pd.DataFrame(df_uniques.groupby(['variable', 'value', 
                                              'cardio'])['value'].count()) \
    .sort_index(level=[0, 1]) \
    .rename(columns={'value': 'count'}) \
    .reset_index()

sns.factorplot(x='variable', y='count', hue='value', 
               col='cardio', data=df_uniques, kind='bar', size=9);
for c in df.columns:
    n = df[c].nunique()
    print(c)
    if n <= 3:
        print(n, sorted(df[c].value_counts().to_dict().items()))
    else:
        print(n)
    print(10 * '-')
df["gender"].value_counts()
df[(df["alco"] == 1) & (df["gender"] == 1)]["gender"].value_counts() / df[df["gender"] == 1]["gender"].value_counts(), df[(df["alco"] == 1) & (df["gender"] == 2)]["gender"].value_counts() / df[df["gender"] == 2]["gender"].value_counts() 

df[(df["smoke"] == 1) & (df["gender"] == 1)]["gender"].value_counts() / df[df["gender"] == 1]["gender"].value_counts(), df[(df["smoke"] == 1) & (df["gender"] == 2)]["gender"].value_counts() / df[df["gender"] == 2]["gender"].value_counts() 

(df[df["smoke"] == 1]["age"].median() - df[df["smoke"] == 0]["age"].median()) / 30
df["age_years"] = np.round(df["age"] / 365, 0)
df["age_years"].describe()

df_old = df[(df["age_years"] >= 60) & (df["age_years"] <= 64)]
df_old["age_years"]

d = {1: "4", 2: "5-7", 3:"8"}
df_old["cholesterol_mmol"] = df_old["cholesterol"].map(d)
df_old["cholesterol_mmol"]
a = df_old[(df_old["cholesterol_mmol"] == "4") & (df_old["ap_hi"] <= 120) & (df_old["cardio"] == 1)].shape[0] / df_old[(df_old["cholesterol_mmol"] == "4") & (df_old["ap_hi"] <= 120)].shape[0]
b = df_old[(df_old["cholesterol_mmol"] == "8") & (df_old["ap_hi"] >= 160) & (df_old["ap_hi"] < 180) & (df_old["cardio"] == 1)].shape[0] / df_old[(df_old["cholesterol_mmol"] == "8") & (df_old["ap_hi"] >= 160) & (df_old["ap_hi"] < 180)].shape[0]

b/a
df["BMI"] = df["weight"] / np.square(df["height"] / 100)
df[["BMI", "height", "weight"]]
df.BMI.median() #1.False
df[df.gender == 1].BMI.mean() > df[df.gender == 2].BMI.mean() #2. True
df[df.cardio == 0].BMI.mean() > df[df.cardio == 1].BMI.mean() #3. False

BMI_distance = lambda x:  x-25 if x > 25 else 18.5-x if x < 18.5 else 0
df["distance"] = df.BMI.map(BMI_distance) #[] notation because of initialization of non-existing column in df

df[(df.cardio == 0) & (df.alco == 0) & (df.gender == 2)].distance.sum() < df[(df.cardio == 0) & (df.alco == 0) & (df.gender == 1)].distance.sum()
df_pre = df
df = df[df.ap_hi >= df.ap_lo]
df = df[(df.height >= df.height.quantile(0.025)) & (df.height <= df.height.quantile(0.975))]
df = df[(df.weight >= df.weight.quantile(0.025)) & (df.weight <= df.weight.quantile(0.975))]

df.shape
df.shape[0] / df_pre.shape[0]
import seaborn as sns

corr = df.corr().round(2)
sns.heatmap(corr, annot=True)
#1 => 0.22
#2 => 0.19
#3 => 0.34
#4 => 0.25
df_long = pd.melt(df, id_vars='id')
sns.violinplot(x=df.cardio, y=df.height, hue=df.gender, data=df, split=True, inner="quartile")
corr_rank = df.corr(method="spearman").round(2)
sns.heatmap(corr_rank, annot=True)
#1 => 0.26
#2 => 0.07
#3 => 0.41
#4 => 0.21
#5 => 0.74
#6 => 0.34
#3
sns.countplot(data=df, x=df.age_years, hue=df.cardio)
#55 years