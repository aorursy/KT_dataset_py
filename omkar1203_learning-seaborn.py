import seaborn as sns
df=sns.load_dataset("tips")
df.head()
df.corr()
sns.heatmap(df.corr())
sns.jointplot(x='tip',y='total_bill',data=df,kind='hex')
sns.jointplot(x='tip',y='total_bill',data=df,kind='reg')
sns.pairplot(df)
sns.pairplot(df,hue='sex')
sns.distplot(df['tip'])
sns.distplot(df['tip'],kde=False,bins=10)
## Count plot



sns.countplot('sex',data=df)
## Count plot



sns.countplot(y='sex',data=df)
## Bar plot

sns.barplot(x='total_bill',y='sex',data=df)
## Bar plot

sns.barplot(x='sex',y='total_bill',data=df)
df.head()
sns.boxplot('smoker','total_bill', data=df)
sns.boxplot(x="day", y="total_bill", data=df,palette='rainbow')
sns.boxplot(data=df,orient='v')
# categorize my data based on some other categories



sns.boxplot(x="total_bill", y="day", hue="smoker",data=df)
sns.violinplot(x="total_bill", y="day", data=df,palette='rainbow')