import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
auto=pd.read_csv("../input/automobiles/Automobile.csv")
auto
auto.head()
auto.tail()
auto.columns
auto.info()
auto[auto.dtypes[auto.dtypes=='object'].index]
plt.figure(figsize=(10,10))
auto['make'].value_counts().plot(kind='bar',color='orange')
plt.show()
plt.figure(figsize=(7,7))
auto['fuel_type'].value_counts(ascending=False).plot(kind='bar',colormap='Paired')
plt.show()
plt.figure(figsize=(7,7))
auto['aspiration'].value_counts().plot(kind='bar',colormap='YlGn_r')
plt.legend()
plt.show()
plt.figure(figsize=(7,7))
auto['number_of_doors'].value_counts().plot(kind='bar',color=['orange','blue'])
plt.show()
plt.figure(figsize=(7,7))
auto['body_style'].value_counts().plot(kind='bar',color=['blue','orange','yellow','red','red'])
plt.show()
fig,axes=plt.subplots(2,2,figsize=[10,8])
fig.suptitle('CategoryPlots',size=15,color='blue')   
auto['drive_wheels'].value_counts().plot(kind='bar',ax=axes[0][0],color='red')
axes[0][0].set_xlabel('Drive wheels',size=10,color='red')
auto['engine_location'].value_counts().plot(kind='bar',ax=axes[0][1],color='orange')
axes[0][1].set_xlabel('Engine_location',size=10,color='red')
auto['engine_type'].value_counts().plot(kind='bar',ax=axes[1][0],color='yellow')
axes[1][0].set_xlabel('Engine_type',size=10,color='red')
auto['number_of_cylinders'].value_counts().plot(kind='bar',ax=axes[1][1])
axes[1][1].set_xlabel('number_of_cylinders',size=10,color='red')
plt.subplots_adjust(hspace=1)
plt.show()
#drive_wheels	engine_location	engine_type	number_of_cylinders
auto[auto.dtypes[auto.dtypes!='object'].index]
sns.distplot(auto['symboling'].dropna(),color="red")
plt.show()
sns.distplot(auto['normalized_losses'].dropna(),color="orange")
plt.show()
sns.distplot(auto['engine_size'].dropna(),color='yellow')
plt.show()
sns.distplot(auto['horsepower'].dropna(),color="green")
plt.show()
sns.distplot(auto['peak_rpm'].dropna())
plt.show()
sns.distplot(auto['price'].dropna(),color="aqua")
plt.show()
auto
sns.scatterplot(x='symboling',y='price',data=auto)
plt.show()
plt.figure(figsize=(10,8))
sns.boxplot(x='make',y='price',data=auto)
plt.xticks(rotation=90)
plt.show()
plt.figure(figsize=(8,8))
sns.boxplot(x='aspiration',y='price',data=auto)                 #aspiration vs price
plt.xticks(rotation=90)
plt.show()
plt.figure(figsize=(8,8))
sns.boxplot(x='number_of_doors',y='price',data=auto)                 #aspiration vs price
plt.xticks(rotation=90)
plt.show()
plotlist=auto.dtypes[auto.dtypes=='object'].index
plotlist
for i in plotlist:
    sns.boxplot(x=i,y='price',data=auto)
    plt.xticks(rotation=90)
    plt.show()
plt.figure(figsize=(10,10))
sns.heatmap(auto.corr(),annot=True)
plt.show()
auto['number_of_doors']=auto['number_of_doors'].replace({'two':2,'four':4})
auto
auto.isna().sum()

plot=auto[auto.dtypes[auto.dtypes!='object'].index] 
plot
for i in plot.columns:
    sns.boxplot(plot[i])
    plt.show()
Q3=auto['normalized_losses'].quantile(0.75)
Q1=auto['normalized_losses'].quantile(0.25)
IQR=Q3-Q1
UP=Q3+1.5*IQR
LB=Q1-1.5*IQR
UP,LB
auto['normalized_losses'].quantile(0.99)
auto[auto['normalized_losses']>UP]['normalized_losses']
box=auto['normalized_losses'].replace(auto[auto['normalized_losses']>UP]['normalized_losses'],auto['normalized_losses'].quantile(0.99))
sns.boxplot(box)
for i in plot.columns:
    Q3=plot[i].quantile(0.75)
    Q1=plot[i].quantile(0.25)
    IQR=Q3-Q1
    UP=Q3+1.5*IQR
    LB=Q1-1.5*IQR
    plot[i]=plot[i].replace(plot[plot[i]>UP][i],plot[i].quantile(0.99))
    plot[i]=plot[i].replace(plot[plot[i]<LB][i],plot[i].quantile(0.01))
    
    
    
    
for i in plot.columns:
    sns.boxplot(plot[i])
    plt.show()
plot
from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
plot_sc=ss.fit_transform(plot)
plot_sc=pd.DataFrame(plot_sc,columns=plot.columns)
plot_sc
plot_sc['symboling'].mean(),plot_sc['symboling'].std()
from sklearn.preprocessing import MinMaxScaler
ms=MinMaxScaler()
plot_ms=ms.fit_transform(plot)
plot_ms=pd.DataFrame(plot_ms,columns=plot.columns)
plot_ms  #Values are lieing between 0 and 1 for min max scaler
plot_sc['symboling'].plot(kind='density',color='aqua')
plt.show()
plot['symboling'].plot(kind='density',color='cadetblue')
plt.show()
from sklearn.preprocessing import RobustScaler
rs=RobustScaler()
plot_rs=rs.fit_transform(plot)
plot_rs=pd.DataFrame(plot_rs,columns=plot.columns)
plot_rs
plot_rs['symboling'].plot(kind='kde')
plt.show()
from sklearn.preprocessing import Normalizer
norm=Normalizer(norm='l2')
plot_norm=norm.fit_transform(plot)
plot_norm=pd.DataFrame(plot_norm,columns=plot.columns)
plot_norm
cat_col=auto.select_dtypes(include='object')
cat_col
cat_col['drive_wheels'].value_counts()
cat_col['fuel_system'].value_counts()
cat_col['fuel_system']=cat_col['fuel_system'].replace({'mpfi':'fi','spfi':'fi','mfi':'fi',
                                                       '1bbl':'bbl','2bbl':'bbl','4bbl':'bbl','idi':'di','spdi':'di'})
cat_col
cat_col['engine_type'].value_counts()
cat_col['engine_type']=cat_col['engine_type'].replace({'dohc':'ohc'})
cat_col['engine_type'].value_counts()
cat_col['body_style'].value_counts()
frequencies=cat_col['make'].value_counts(normalize=True)
cat_col['make']=cat_col['make'].map(frequencies)
cat_col
cat_col['fuel_system'].value_counts()
cat_dum=pd.get_dummies(cat_col,columns=['fuel_type','aspiration','body_style','drive_wheels','engine_location','engine_type','number_of_cylinders','fuel_system'],drop_first=True)
cat_dum
final_data=pd.concat([plot_sc,cat_dum],axis=1)
final_data
final_data.shape
final_data.columns
auto.shape
from sklearn.model_selection import train_test_split
out=final_data['price']
inp=final_data.drop('price',axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(inp,out,test_size=0.3,random_state=0)
print(xtrain.shape)
print(xtest.shape)
print(ytest.shape)
print(ytrain.shape)
