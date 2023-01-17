# imports
import altair as alt
from altair import datum
import pandas as pd
import numpy as np
import matplotlib 
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
# alt.renderers.enable('notebook')
# Load the bike sharing data as a 'bike'
bike = pd.read_csv('../input/bike-sharing-dataset/day.csv')
#Data Analysis
plt.figure(figsize=(20,5))
mask = np.zeros_like(bike.corr(), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(bike.corr(),cmap='RdBu_r',mask=mask, annot=True)
# Our heat data is normalised data( The values are derived via (t-t_min)/(t_max-t_min), t_min=-8, t_max=+39). For readability and visualisation we multiply it by 100
bike['temp'] = bike['temp']*100
# Also we have many decimal numbers in the data set so we round off to 1 decimal place
bike = bike.round(1)
bike['holiday'][bike['holiday'] == 0] = 'Not Holiday'
bike['holiday'][bike['holiday'] == 1] = 'Holiday'
# Change the values of the categorical data to improve the readability on visualization. 
bike['holiday'][bike['holiday'] == 0] = 'Not Holiday'
bike['holiday'][bike['holiday'] == 1] = 'Holiday'
bike['weathersit'][bike['weathersit'] == 1] = 'Clear'
bike['weathersit'][bike['weathersit'] == 2] = 'Cloudy'
bike['weathersit'][bike['weathersit'] == 3] = 'Snowy'
bike['yr'][bike['yr'] == 0] = '2011'
bike['yr'][bike['yr'] == 1] = '2012'
bike['season'][bike['season'] == 1] = 'winter'
bike['season'][bike['season'] == 2] = 'spring'
bike['season'][bike['season'] == 3] = 'summer'
bike['season'][bike['season'] == 4] = 'fall'
# We drop the column 'instant' because it is useless.
bike.drop('instant', axis = 1, inplace = True)
# We create new column 'tem_range' by using pandas cut method. 4 equal interval is created from temperature feature.
bike['temp_range'] = pd.cut(x=bike['temp'], bins=[5, 25, 45, 65,87], labels=['5-24', '25-45', '46-65','66-87'])
# Our first five samples
bike.head()
bike.describe()
bike.dtypes
#Check the missing value
sns.heatmap(bike.isnull(), yticklabels = False, cbar = False, cmap = 'viridis')
# Fortunately we have not missing value
# We should create dummy variables because regression can't understand categorical variable
season = pd.get_dummies(bike['season'],drop_first = True)
year = pd.get_dummies(bike['yr'],drop_first = True)
weather = pd.get_dummies(bike['weathersit'],drop_first = True)
bike = pd.concat([bike,season,year,weather],axis =1)
bike
bike.columns
y = bike['cnt']
X = bike[['spring','summer','winter','2012','temp','Cloudy','Snowy']]
# We take selected features as a parameter based on my task
# We divided the X and y into train and test data
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.4, random_state = 101)
lm = LinearRegression()
lm.fit(X_train,y_train)
pd.DataFrame(lm.coef_,X.columns,columns =['Coeff'])
pred = lm.predict(X_test)
mse = np.mean((pred - y_test)**2)
print(lm.score(X_test,y_test))                  #R^2 value of Linear Regression
mse                                             # MSE value of Linear Regression
plt.scatter(y_test,pred)
plt.xlabel('Acutal number of shared bikes')
plt.ylabel('Predicted number of sahred bikes')
df = pd.DataFrame({'Actual': y_test.tolist(), 'Predicted': pred.tolist()})
df
sns.regplot(x= 'temp',y= 'cnt', data=bike)
plt.xlabel('Normalized temperature')
plt.ylabel('Number of shared bikes')
plt.title('Normalized temperature vs Number of shared bikes')
sns.barplot(x="season", y="cnt", hue="yr", data=bike)
plt.xlabel('Seasons')
plt.ylabel('Number of shared bikes')
plt.title('Seasons vs Number of Shared Bikes')
# Lasso Regression
lasso = Lasso(alpha=0.3, normalize=True)
lasso.fit(X_train,y_train)
pred = lasso.predict(X_test)
mse = np.mean((pred - y_test)**2)
print(lasso.score(X_test,y_test))
mse
Yr = ['2011', '2012'] 
select = alt.selection_single(name = 'Select',
                              fields=['yr'],init={'yr': Yr[0]},
                              bind= alt.binding_radio(options = Yr))
pts = alt.selection(type="single", encodings=['x'])
brush = alt.selection_interval(encodings=['x'],empty='all')
base = alt.Chart(bike).encode(
    alt.X('month(dteday):T',title = ' Months '),
    alt.Y('mean(cnt):Q',scale = alt.Scale(domain = [0,9000],round = True),title = 'Mean Number of Shared Bikes')
).properties(
    width=450,
    height=400,title = 'Months of the Selected Year vs Mean Number of Shared Bikes' 
)
scale = alt.Scale(domain=['5-24', '25-45', '46-65','66-87'],
                  range=['#636363', '#3182bd','#ff7f0e','#e41a1c'])
color = alt.Color('temp_range:N', scale=scale)
bar = base.mark_bar(size =20).add_selection(select).encode(
color = alt.condition(pts, alt.ColorValue("steelblue"), 
                      alt.ColorValue("grey")),
tooltip = [alt.Tooltip('mean(cnt):Q',title = 'Count of average rental bikes')]).transform_filter(select).add_selection(pts)
line = alt.Chart(bike).mark_line(color = 'grey').encode(
    alt.X('date(dteday):O',title = 'Days'),
    alt.Y('cnt:Q',scale = alt.Scale(domain = [0,9000]),title = 'Number of Shared Bikes')).transform_filter(select).transform_filter(pts).properties(
    width=400,
    height=400,title = 'Days of the Selected Month vs Number of Shared Bikes'
)

points = alt.Chart(bike).mark_point(filled = True,size = 75).add_selection(brush).encode(alt.X('date(dteday):O'),
    alt.Y('cnt:Q'),
tooltip = [alt.Tooltip('cnt:Q',title = 'Count of rental bikes'),alt.Tooltip('temp:Q')],
color = alt.condition(brush,color,alt.value('lightgray'))).transform_filter(select).transform_filter(pts)
bar_weather = alt.Chart(bike).mark_bar().encode(
    y='weathersit:N',
    x='count():Q',
).transform_filter(select).transform_filter(pts).transform_filter(brush)
text = bar_weather.mark_text(
    align='left',
    baseline='middle',
    dx=3  
).encode(text = 'count(weathersit):Q')


rule = alt.Chart(bike).mark_rule(color = 'black').encode(
    y='mean(cnt)',
    size=alt.value(2)).transform_filter(select).transform_filter(pts)
rule2 = alt.Chart(bike).mark_rule(color = 'black').encode(
    y='mean(cnt)',
    size=alt.value(2)).transform_filter(select)


bar + rule2 | ((line + points) + rule) & bar_weather + text

label = alt.selection_single(
    encodings=['x'],  
    on='mouseover',  
    nearest=True,     
    empty='none'     
)
pts = alt.selection(type="single", encodings=['x'])

base = alt.Chart().mark_line().encode(
    alt.X('mnth:N',title = 'Months'),
    alt.Y('mean(cnt):Q',scale = alt.Scale(domain = [0,9000]),title = 'Mean Number of Shared Bikes'),
    alt.Color('yr:N')
).properties(width = 450,height = 400,title = 'Months vs Mean Number of Shared Bikes')

x = alt.layer(
    base, 
    alt.Chart().mark_rule(color='#aaa').encode(
        x='mnth:N'
    ).transform_filter(label),
    base.mark_circle(filled = True,size = 200).add_selection(pts).encode(
        opacity=alt.condition(label, alt.value(1), alt.value(0)),
        color = alt.condition(pts,'yr:N',alt.value('lightgray'))
    ).add_selection(label),
    base.mark_text(align='left', dx=5, dy=-5, stroke='white', strokeWidth=2).encode(
        text='mean(cnt):Q'
    ).transform_filter(label),
    base.mark_text(align='left', dx=5, dy=-5).encode(
        text='mean(cnt):Q' ).transform_filter(label),
     data=bike
).properties(
    width=700,
    height=400
)
base2 = alt.Chart().mark_line().encode(
    alt.X('date(dteday)', title = 'Days'),
    alt.Y('cnt:Q',scale = alt.Scale(domain = [0,9000]),title = 'Number of Shared Bikes'),
    alt.Color('yr:N')
).properties(width =400,height=400,title = 'Days of the Selected Month vs Number of Shared Bikes')

y = alt.layer(
    base2, 
    alt.Chart().mark_rule(color='#aaa').encode(
     x = 'date(dteday)'
    ).transform_filter(label),   
    base2.mark_circle(filled = True,size = 200).add_selection(pts).encode(
        opacity=alt.condition(label, alt.value(1), alt.value(0))
    ).add_selection(label),
    base2.mark_text(align='left', dx=5, dy=-5, stroke='white', strokeWidth=2).encode(
        text='cnt:Q'
    ).transform_filter(label),
    base2.mark_text(align='left', dx=5, dy=-5).encode(
        text='cnt:Q' ).transform_filter(label),
     data=bike
).properties(
    width=700,
    height=400
).transform_filter(pts)

base3 = alt.Chart().mark_line().encode(
    alt.X('date(dteday)',title = 'Days'),
    alt.Y('temp:Q',scale = alt.Scale(domain = [0,90]),title = 'Normalized Temperature'),
    alt.Color('yr:N')).properties(width =400,height=400,title = 'Days of the Selected Month vs Normalized Temperature')

z = alt.layer(
    base3, 
    alt.Chart().mark_rule(color='#aaa').encode(
     x = 'date(dteday)'
    ).transform_filter(label),
    base3.mark_circle(filled = True,size = 200).add_selection(pts).encode(
        opacity=alt.condition(label, alt.value(1), alt.value(0))
    ).add_selection(label),
    base3.mark_text(align='left', dx=5, dy=-5, stroke='white', strokeWidth=2).encode(
        text='temp:Q'
    ).transform_filter(label),
    base3.mark_text(align='left', dx=5, dy=-5).encode(
        text='temp:Q' ).transform_filter(label),
     data=bike
).properties(
    width=700,
    height=400
).transform_filter(pts)
x|y&z

