import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
%matplotlib inline
import random
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.colors import n_colors
from plotly.subplots import make_subplots
init_notebook_mode(connected=True)
import cufflinks as cf
cf.go_offline()
from wordcloud import WordCloud , ImageColorGenerator
from PIL import Image
df = pd.read_csv('../input/indian-food-101/indian_food.csv')
df=df.replace(-1,np.nan)
df=df.replace('-1',np.nan)
df.head()
df.shape
pie_df = df.diet.value_counts().reset_index()
pie_df.columns = ['diet','count']
fig = px.pie(pie_df, values='count', names='diet', title='Proportion of Vegetarian and Non-Vegetarian dishes',
             color_discrete_sequence=['green', 'red'])
fig.show()
reg_df = df.region.value_counts().reset_index()
reg_df.columns = ['region','count']
reg_df = reg_df.sample(frac=1)
fig = px.bar(reg_df,x='region',y='count',title='Number of dishes based on regions',
             color_discrete_sequence=['#316394'])
fig.show()
shp_gdf = gpd.read_file('../input/india-gis-data/India States/Indian_states.shp')
desserts = df[df['course']=='dessert']
des_df = desserts.state.value_counts().reset_index()
des_df.columns = ['state','count']
merged = shp_gdf.set_index('st_nm').join(des_df.set_index('state'))
fig, ax = plt.subplots(1, figsize=(10, 10))
ax.axis('off')
ax.set_title('State-wise Distribution of Indian Sweets',
             fontdict={'fontsize': '15', 'fontweight' : '3'})
fig = merged.plot(column='count', cmap='Wistia', linewidth=0.5, ax=ax, edgecolor='0.2',legend=True)
course_df = df.course.value_counts().reset_index()
course_df.columns = ['course','count']
course_df = course_df.sample(frac=1)
fig = px.bar(course_df,x='course',y='count',title='Number of dishes based on courses of meal',
             color_discrete_sequence=['#AB63FA'])
fig.show()
pie_df = df.flavor_profile.value_counts().reset_index()
pie_df.columns = ['flavor','count']
fig = px.pie(pie_df, values='count', names='flavor', title='Proportion of Flavor Profiles',
             color_discrete_sequence=['#FF7F0E', '#00B5F7','#AB63FA','#00CC96'])
fig.show()
dessert_df  = df[df['course']=='dessert'].reset_index()

ingredients = []
for i in range(0,len(dessert_df)):
    text = dessert_df['ingredients'][i].split(',')
    text = ','.join(text)
    ingredients.append(text)
    text = ' '.join(ingredients)

wordcloud = WordCloud(width = 400, height = 400, colormap = 'seismic'
                      ,background_color ='white', 
                min_font_size = 10).generate(text)                  
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis('off') 
plt.show()
south_df = df[df['region']=='South'].reset_index()

ingredients = []
for i in range(0,len(south_df)):
    text = south_df['ingredients'][i].split(',')
    text = ','.join(text)
    ingredients.append(text)
    text = ' '.join(ingredients)
    
wordcloud = WordCloud(width = 400, height = 400, colormap = 'spring'
                      ,background_color ='white', 
                min_font_size = 10).generate(text)                  
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis('off') 
plt.show()
sweet_df = df[df['flavor_profile']=='sweet']
final_sweet_df = sweet_df[sweet_df['course']!='dessert']
#final_sweet_df
north_df = df[df['region']=='North'].reset_index()

ingredients = []
for i in range(0,len(north_df)):
    text = north_df['ingredients'][i].split(',')
    text = ','.join(text)
    ingredients.append(text)
    text = ' '.join(ingredients)

wordcloud = WordCloud(width = 400, height = 400, colormap = 'winter',
                      background_color ='white', 
                min_font_size = 10).generate(text)                  
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis('off') 
plt.show()
ingredients = []
for i in range(0,len(df)):
    text = df['ingredients'][i].split(',')
    text = ','.join(text)
    ingredients.append(text)
    text = ' '.join(ingredients)

india_coloring = np.array(Image.open('../input/images/ind.jpg'))

wc = WordCloud(background_color="black", width = 400, height = 400,mask=india_coloring,min_font_size=8)
wc.generate(text)

image_colors = ImageColorGenerator(india_coloring)

plt.figure(figsize = (8, 8))
plt.imshow(wc.recolor(color_func=image_colors), interpolation="bilinear")
plt.axis('off')
plt.show()
fig = px.scatter(df,x='cook_time',y='prep_time',color='diet',
                 color_discrete_sequence=['green','red'], hover_data = ['name'],
                 labels={
                     'cook_time': 'Cooking time (minutes)',
                     'prep_time': 'Preparation time (minutes)'
                 })
fig.show()
mah_df = df[df['state']=='Maharashtra']

total_dishes = mah_df.shape[0]

course_df = mah_df['course'].value_counts().reset_index()
course_df.columns = ['course','num']

diet_df = mah_df['diet'].value_counts().reset_index()
diet_df.columns = ['diet','num']

fig = make_subplots(
    rows=2, cols=2,subplot_titles=('Total Dishes','Dishes by Courses','', ''),
    specs=[[{'type': 'indicator'},{'type': 'bar','rowspan': 2} ],
          [ {'type': 'pie'} , {'type': 'pie'}]])

fig.add_trace(go.Indicator(
    mode = 'number',
    value = int(total_dishes),
    number={'font':{'color': 'blue','size':50}},
),row=1, col=1)


fig.add_trace(go.Bar(
    x=course_df['course'],y=course_df['num'],
    marker={'color': 'orange'},  
    text=course_df['num'],
    name='dishes by courses',
    textposition ='auto'),row=1, col=2)

fig.add_trace(go.Pie(labels=diet_df['diet'], 
                     values=diet_df['num'],textinfo='percent',
                    marker= dict(colors=['green','red'])),row=2, col=1)


fig.update_layout(
    title_text='Maharashtra Food-Mini Infograph',template='plotly',
    title_x=0.5)

fig.show()
def green_color_func(word, font_size, position, orientation, random_state=None,
                    **kwargs):
    return "hsl(100, 100%%, %d%%)" % random.randint(20, 40)


veg_df = df[df['diet']=='vegetarian'].reset_index()

ingredients = []
for i in range(0,len(veg_df)):
    text = veg_df['ingredients'][i].split(',')
    text = ','.join(text)
    ingredients.append(text)
    text = ' '.join(ingredients)

wordcloud = WordCloud(width = 400, height = 400, background_color ='white', 
                min_font_size = 10).generate(text)                  
plt.figure(figsize = (8, 8), facecolor = None) 

default_colors = wordcloud.to_array()
plt.imshow(wordcloud.recolor(color_func=green_color_func, random_state=3),
           interpolation="bilinear")

plt.axis('off') 
plt.show()
def red_color_func(word, font_size, position, orientation, random_state=None,
                    **kwargs):
    return "hsl(0, 100%%, %d%%)" % random.randint(30, 50)


nveg_df = df[df['diet']=='non vegetarian'].reset_index()

ingredients = []
for i in range(0,len(nveg_df)):
    text = nveg_df['ingredients'][i].split(',')
    text = ','.join(text)
    ingredients.append(text)
    text = ' '.join(ingredients)

wordcloud = WordCloud(width = 400, height = 400, background_color ='white', 
                min_font_size = 10).generate(text)                  
plt.figure(figsize = (8, 8), facecolor = None) 

default_colors = wordcloud.to_array()
plt.imshow(wordcloud.recolor(color_func=red_color_func, random_state=3),
           interpolation="bilinear")

plt.axis('off') 
plt.show()
snack_df = df[df['course']=='snack']

short_sort_snack_df = snack_df.sort_values(['cook_time'],ascending=True).iloc[:10,:]

fig = px.bar(short_sort_snack_df,y='name',x='cook_time',
             orientation='h',color='cook_time',
            labels={'name':'Name of snack','cook_time':'Cooking time (minutes)'})
fig.show()
long_sort_snack_df = snack_df.sort_values(['cook_time'],ascending=True).iloc[26:36,:]

fig = px.bar(long_sort_snack_df,y='name',x='cook_time',
             orientation='h',color='cook_time',
            labels={'name':'Name of snack','cook_time':'Cooking time (minutes)'})
fig.show()
mc_df = df[df['course']=='main course']

small_mc_df = mc_df.sort_values(['cook_time'],ascending=True).iloc[:10,:]
fig = px.bar(small_mc_df,y='name',x='cook_time',
             orientation='h',color='cook_time',
            labels={'name':'Name of main course','cook_time':'Cooking time (minutes)'})
fig.show()
long_mc_df = mc_df.sort_values(['cook_time'],ascending=True).iloc[-30:-20,:]

fig = px.bar(long_mc_df,y='name',x='cook_time',
             orientation='h',color='cook_time',
            labels={'name':'Name of main course','cook_time':'Cooking time (minutes)'})
fig.show()