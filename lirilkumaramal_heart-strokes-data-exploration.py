import pandas as pd
import missingno
import seaborn as sns
import matplotlib.pyplot as plt
data_folder = r"/kaggle/input/heart-stroke/"
strokes_df = pd.read_csv(data_folder+r"train_strokes.csv", index_col="id")
strokes_df[['gender','ever_married','Residence_type','work_type','smoking_status','hypertension','stroke']] = strokes_df[['gender','ever_married','Residence_type','work_type','smoking_status','hypertension','stroke']].astype('category')
strokes_df.shape
strokes_df.info()
# here we have
categorical_vars = ['gender','hypertension','heart_disease', 'ever_married','Residence_type', 'work_type', 'smoking_status', 'stroke']
continuous_vars = ['age','avg_glucose_level','bmi']
strokes_df.head()
strokes_df[continuous_vars].describe()
missingno.matrix(strokes_df, figsize = (30,5))
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1,4,figsize=(25,7))

fig.suptitle("Countplot for strokes_df", fontsize=35)

sns.countplot(x="gender", data=strokes_df,ax=ax1)
sns.countplot(x="stroke", data=strokes_df,ax=ax2)
sns.countplot(x="ever_married", data=strokes_df,ax=ax3)
sns.countplot(x="hypertension", data=strokes_df,ax=ax4)
fig, (ax1, ax2, ax3) = plt.subplots(1,3,figsize=(25,7))

fig.suptitle("Countplot for strokes_df", fontsize=35)

sns.countplot(x="work_type", data=strokes_df,ax=ax1)
sns.countplot(x="Residence_type", data=strokes_df,ax=ax2)
sns.countplot(x="smoking_status", data=strokes_df,ax=ax3)

g = sns.catplot(x="Residence_type", hue="smoking_status", col="work_type",
                data=strokes_df, kind="count",
                height=4, aspect=.7)
sns.histplot(strokes_df[continuous_vars], kde=True)
sns.displot(x="age", data=strokes_df, kind="kde", hue="gender", col="smoking_status", row="Residence_type")
fig, (ax1, ax2, ax3) = plt.subplots(1,3,figsize=(20,7))
fig.suptitle("Boxplot for Strokes", fontsize=35)

sns.boxplot(x="stroke", y="avg_glucose_level", data=strokes_df,ax=ax1)
sns.boxplot(x="stroke", y="bmi", data=strokes_df,ax=ax2)
sns.boxplot(x="stroke", y="age", data=strokes_df,ax=ax3)
temp = strokes_df
temp['stroke'] = temp['stroke'].astype(int)
residence_type_df = strokes_df.groupby(["Residence_type"])['stroke'].agg(['sum','count']).reset_index()
residence_type_df['risk'] = residence_type_df['sum'] / residence_type_df['count'] * 100
residence_type_df
temp = strokes_df
temp['stroke'] = temp['stroke'].astype(int)
work_type_df = strokes_df.groupby(["work_type"])['stroke'].agg(['sum','count']).reset_index()
work_type_df['risk'] = work_type_df['sum'] / work_type_df['count'] * 100
work_type_df
temp = strokes_df
temp['stroke'] = temp['stroke'].astype(int)
smoking_status_df = temp.groupby(["smoking_status"])['stroke'].agg(['sum','count']).reset_index()
smoking_status_df['risk'] = smoking_status_df['sum'] / smoking_status_df['count'] * 100
smoking_status_df
temp = strokes_df
temp['stroke'] = temp['stroke'].astype(int)
ever_married_df = temp.groupby(["ever_married"])['stroke'].agg(['sum','count']).reset_index()
ever_married_df['risk'] = ever_married_df['sum'] / ever_married_df['count'] * 100
ever_married_df
temp = strokes_df
temp['stroke'] = temp['stroke'].astype(int)
ever_married_df = temp.groupby(["ever_married","smoking_status"])['stroke'].agg(['sum','count']).reset_index()
ever_married_df['risk'] = ever_married_df['sum'] / ever_married_df['count'] * 100
ever_married_df
sns.boxplot(x="stroke", y="bmi",data=strokes_df)
plt.title("! Feature idea : does BMI over 60 means no stroke?")
sns.boxplot(x="gender", y="bmi", hue="stroke",data=strokes_df)
plt.title("Thought: No stroke if gender is Other?")
sns.boxplot(x="stroke", y="age", hue="gender",data=strokes_df)
plt.title("Anomaly: High age, more risk, few young cases where young person have stroke")
sns.set(rc={'figure.figsize':(17,5)})
sns.boxplot(x="work_type", y="avg_glucose_level", hue="smoking_status",data=strokes_df)
plt.title("Nover worked and Smoker is very high under risk")
sns.set(rc={'figure.figsize':(17,5)})
sns.boxplot(x="work_type", y="avg_glucose_level", hue="stroke",data=strokes_df)
plt.title("Nover worked and Smoker is very high under risk")
g = sns.FacetGrid(strokes_df, col="heart_disease", hue="stroke", height=4, aspect=1.6, row="work_type")
g.map(sns.scatterplot, "avg_glucose_level", "age", alpha=.7)
g.add_legend()
sns.displot(data=strokes_df, x="age", hue="stroke", multiple="stack", kind="kde", col="work_type", row="smoking_status")
plt.title("Stroke and No Stroke observation per combination")
sns.pairplot(data=strokes_df[continuous_vars+["stroke"]], hue="stroke")
g = sns.FacetGrid(strokes_df, col="work_type", hue="gender", height=4, aspect=1.6,row="ever_married")
g.map(sns.countplot, "smoking_status", alpha=.7)
g.add_legend()
strokes_df.columns
import researchpy as rp
rp.summary_cont(strokes_df[['avg_glucose_level','age']].groupby(strokes_df['stroke']))
# Compute a correlation matrix and convert to long-form
corr_mat = strokes_df.corr("kendall").stack().reset_index(name="correlation")

# Draw each cell as a scatter point with varying size and color
g = sns.relplot(
    data=corr_mat,
    x="level_0", y="level_1", hue="correlation", size="correlation",
    palette="vlag", hue_norm=(-1, 1), edgecolor=".7",
    height=5, sizes=(50, 250), size_norm=(-.2, .8),
)

# Tweak the figure to finalize
g.set(xlabel="", ylabel="", aspect="equal")
g.despine(left=True, bottom=True)
g.ax.margins(0.25)
for label in g.ax.get_xticklabels():
    label.set_rotation(90)
for artist in g.legend.legendHandles:
    artist.set_edgecolor(".1")
strokes_temp_df=strokes_df
strokes_temp_df[['stroke','hypertension']] = strokes_df[['stroke','hypertension']].astype('int')
corr = strokes_temp_df.corr()
corr.style.background_gradient()
corr.style.background_gradient().set_precision(2)