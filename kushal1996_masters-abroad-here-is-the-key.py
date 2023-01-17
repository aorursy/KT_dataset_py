import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import plotly as py 
import plotly.graph_objs as go
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from IPython.display import HTML
import os
import warnings

plt.style.use('fivethirtyeight')
py.offline.init_notebook_mode(connected = True)
warnings.filterwarnings("ignore")
HTML('''
<script>
  function code_toggle() {
    if (code_shown){
      $('div.input').hide('500');
      $('#toggleButton').val('Show Code')
    } else {
      $('div.input').show('500');
      $('#toggleButton').val('Hide Code')
    }
    code_shown = !code_shown
  }

  $( document ).ready(function(){
    code_shown=false;
    $('div.input').hide()
  });
</script>
<form action="javascript:code_toggle()"><input type="submit" id="toggleButton" value="Show Code"></form>''')
gre = pd.read_csv(r'../input/Admission_Predict_Ver1.1.csv')
gre.head()
gre.isnull().sum()
gre.dtypes
gre.describe()
df = pd.DataFrame.copy(gre)
del df['Serial No.']
df.head()
plt.figure(1 , figsize = (15 , 7))
df['GRE Score'].plot(kind = 'hist' , bins = 50)
plt.title('Histogram of GRE Score')
plt.show()
print('Minimum GRE score : {} , Average GRE score : {} and Maximum GRE score : {}'.format(
        df['GRE Score'].min() , df['GRE Score'].mean() , df['GRE Score'].max()))
plt.figure(1 , figsize = (15 , 7))
df['TOEFL Score'].plot(kind = 'hist' , bins = 15 )
plt.title('Histogram of TOEFL score')
plt.show()
print('Minimum TOEFL score : {} , Average TOEFL score : {} and Maximum TOEFL score : {}'.format(
    df['TOEFL Score'].min() , df['TOEFL Score'].mean() , df['TOEFL Score'].max()))
plot_val = ['SOP' ,'LOR ']
plt.figure(1 , figsize = (15 , 7 ))
n = 0
for i in plot_val:
    n += 1
    plt.subplot(1 , 2 , n)
    df['SOP'].plot(kind = 'hist')
    plt.title('Histogram of '+i+'strenght')
plt.show()
print('Average SOP rating : {} , Average TOEFL rating : {}'.format(
    df['SOP'].mean() , df['LOR '].mean()))
plt.figure(1, figsize = (15 , 7))
sns.countplot(x = 'Research' , data =  df , palette = 'rocket')
plt.title('Count plot of Research experience (0 or 1)')
plt.show()
plt.figure(1 , figsize = (15 , 7) )
sns.countplot(y = 'University Rating' , data  = df , palette = 'rocket')
plt.title('Count plot of University rating')
plt.show()
plt.figure(1 , figsize =  (15 , 7))
sns.distplot(a = df['Chance of Admit '] , bins = 59 , color = 'r')
plt.title('Distplot of Chance of Admit probability')
plt.show()
cor = df.corr()
plt.figure(1 , figsize = (15 , 8))
sns.heatmap(cor , annot = True )
plt.title('Heat map')
plt.show()
g = sns.PairGrid(df)
g.map_diag(sns.kdeplot)
g.map_offdiag(sns.kdeplot, n_levels=6);
plot_val  = ['GRE Score' , 'TOEFL Score' , 'Chance of Admit ']
plt.figure(1,  figsize  = (15 , 12))
n = 0 
for i in plot_val:
    n += 1 
    plt.subplot(3 , 1 , n )
    sns.regplot(x = 'CGPA' , y = i , data = df)
plt.show()
plt.figure(1 , figsize = (15 , 6))
for n in [0.5 , 0.8]:
    plt.plot(df['CGPA'],np.ones((df.shape[0] , 1))*n,'r-' , alpha = 0.5)

plt.scatter(x = 'CGPA' ,y = 'Chance of Admit ' , data = df.where(df['Chance of Admit '] >= 0.80),
           s = 100 , marker = 'o' , c = 'green')
plt.scatter(x = 'CGPA' ,y = 'Chance of Admit ' , data = df.where(df['Chance of Admit '] < 0.80),
           s = 100 , marker = 'o', c = 'blue')
plt.scatter(x = 'CGPA' ,y = 'Chance of Admit ' , data = df.where(df['Chance of Admit '] < 0.51),
           s = 100 , marker = 'o', c = 'red')
plt.xlabel('CGPA')
plt.ylabel('Chance of admit probability')
plt.show()
plt.figure(1 , figsize = (15 , 7))
plt.plot(np.ones((df.shape[0] , 1))*8 , df['GRE Score'] , '-' , alpha = 0.5 )
plt.scatter(x = 'CGPA' , y = 'GRE Score' , data = df.where(df['CGPA'] >=8) , s = 200 ,
           marker = '+')
plt.scatter(x = 'CGPA' , y = 'GRE Score' , data = df.where(df['CGPA'] < 8) ,s =  200,
           marker = 'o')
plt.xlabel('CGPA')
plt.ylabel('GRE Score')
plt.show()
plt.figure(1 , figsize = (15 , 7))
plt.plot(np.ones((df.shape[0] , 1))*330 , df['TOEFL Score'] , '-' , alpha = 0.5 )
plt.scatter(x = 'GRE Score' , y = 'TOEFL Score' , data = df.where(df['GRE Score'] >=330) ,
            s = 200 , marker = '+')
plt.scatter(x = 'GRE Score' , y = 'TOEFL Score' , data = df.where(df['GRE Score'] < 330) ,
            s =  200,marker = 'o')
plt.xlabel('GRE Score')
plt.ylabel('TOEFL Score')
plt.show()

plt.figure(1 , figsize = (15 , 6))
for n in [0.5 , 0.8]:
    plt.plot(df['GRE Score'],np.ones((df.shape[0] , 1))*n,'r-' , alpha = 0.5)
plt.scatter(x = 'GRE Score' , y = 'Chance of Admit ' , data = df.where(df['Chance of Admit '] >= 0.8),
           c = 'green' , s = 200)
plt.scatter(x = 'GRE Score' , y = 'Chance of Admit ' , data = df.where(df['Chance of Admit '] < 0.8),
            s = 200)
plt.scatter(x = 'GRE Score' , y = 'Chance of Admit ' , data = df.where(df['Chance of Admit '] < 0.5),
            s = 200)

plt.xlabel('GRE Score')
plt.ylabel('Chance of Admit')
plt.show()
plt.figure(1 , figsize = (15 , 6))
for n in [0.5 , 0.8]:
    plt.plot(df['TOEFL Score'],np.ones((df.shape[0] , 1))*n,'r-' , alpha = 0.5)
plt.scatter(x = 'TOEFL Score' , y  = 'Chance of Admit ' ,
            data = df.where(df['Chance of Admit '] >= 0.8) , s = 200 , c = 'green')
plt.scatter(x = 'TOEFL Score' , y  = 'Chance of Admit ' ,
            data = df.where(df['Chance of Admit '] < 0.8) , s = 200)
plt.scatter(x = 'TOEFL Score' , y  = 'Chance of Admit ' ,
            data = df.where(df['Chance of Admit '] < 0.5) , s = 200 )
plt.xlabel('TOEFL Score')
plt.ylabel('Chance of Admit')
plt.show()
plt.figure(1 , figsize = (15 , 6))
plt_val  = ['GRE Score' , 'TOEFL Score']
n = 0
for i in plt_val:
    n += 1
    plt.subplot(1 , 2 , n)
    z = (df[i].quantile(0.9))
    plt.scatter(x = 'University Rating' , y = i , data = df.where(df[i] >= z) ,
                s = 200 , c = 'green')
    plt.scatter(x = 'University Rating' , y = i , data = df.where(df[i] < z) ,
                s = 200 )
    plt.plot(df['University Rating'] , np.ones((df.shape[0] , 1 ))*z , '-' , alpha = 0.5)
    plt.xlabel('University Rating')
    plt.ylabel(i)
plt.show()

plt.figure(2 , figsize  = (15 , 6))

plt.plot(df['University Rating'] , np.ones((df.shape[0] , 1))*8 , '-' , alpha = 0.5 )
plt.scatter(x = 'University Rating' , y = 'CGPA' ,
            data = df.where(df['CGPA'] >= 8) , s = 200 , c = 'green' )
plt.scatter(x = 'University Rating' , y = 'CGPA' ,
            data = df.where(df['CGPA'] < 8) , s = 200  )

plt.xlabel('University Rating')
plt.ylabel('CGPA')
plt.show()


plt.figure(3 , figsize  = (15 , 6))
for n in [0.8 , 0.5]:
    plt.plot(df['University Rating'] , np.ones((df.shape[0] , 1))*n , '-' , alpha = 0.5 )
plt.scatter(x = 'University Rating' , y = 'Chance of Admit ' ,
            data = df.where(df['Chance of Admit '] >= 0.8) , s = 200 , c = 'green' )
plt.scatter(x = 'University Rating' , y = 'Chance of Admit ' ,
            data = df.where(df['Chance of Admit '] < 0.8) , s = 200  )
plt.scatter(x = 'University Rating' , y = 'Chance of Admit ' ,
            data = df.where(df['Chance of Admit '] < 0.5) , s = 200  )
plt.xlabel('University Rating')
plt.ylabel('Chance of Admit')
plt.show()

plt.figure(1 , figsize = (15 , 7))
n = 0
for i in ['SOP' , 'LOR ']:
    n += 1
    plt.subplot(1 , 2 , n)
    sns.countplot( y = i , data = df , palette = 'rocket')
plt.show()
plt.figure(1 , figsize = (15 , 7))
n = 0 
for i in ['SOP' , 'LOR ']:
    n += 1
    plt.subplot(1 , 2 , n)
    for z in [0.8 , 0.5]:
        plt.plot(np.ones((df.shape[0] , 1))*z , df[i] , '-' , alpha = 0.5)
    plt.scatter(x = 'Chance of Admit ' , y = i ,
                data = df.where(df['Chance of Admit '] >= 0.8 ) , s = 200 , c = 'green')
    plt.scatter(x = 'Chance of Admit ' , y = i ,
                data = df.where(df['Chance of Admit '] < 0.8 ) , s = 200)
    plt.scatter(x = 'Chance of Admit ' , y = i ,
                data = df.where(df['Chance of Admit '] < 0.5 ) , s = 200)
    plt.xlabel('Chance of Admit')
    plt.ylabel(i)
plt.show()
final_features_df = pd.DataFrame(df ,columns = ['GRE Score' , 'TOEFL Score' , 'CGPA' ,'SOP' , 'LOR ',
                                       'Chance of Admit '])
x = final_features_df.iloc[: , :-1].values
y = final_features_df.iloc[: , -1].values
y = y.reshape([len(y) , 1])

x_train , x_test , y_train , y_test = train_test_split(x , y , 
                                                      test_size = 0.3, 
                                                      random_state = 111)

from sklearn.linear_model import LinearRegression

algo = (LinearRegression())
algo.fit(x_train , y_train)
y_pred = algo.predict(x_test)

#print('r2 score {}'.format(r2_score(y_test , y_pred)))