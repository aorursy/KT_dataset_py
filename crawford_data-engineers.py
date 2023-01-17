import altair as alt

alt.renderers.enable('kaggle')



import pandas as pd

pd.set_option('max_colwidth', 300)

pd.set_option('max_columns', None)



import re
df = pd.read_csv("../input/kaggle-survey-2019/multiple_choice_responses.csv")

ques = pd.read_csv("../input/kaggle-survey-2019/questions_only.csv")

ques.T
data = df.drop(df.index[0]).groupby('Q5')['Q5'].size()



data = pd.DataFrame({

    'title': data.index,

    'count': data.values

})



bars = alt.Chart(data).mark_bar().encode(

    x=alt.X('count:Q', title='', axis=None),

    y=alt.Y('title:N', title='', sort=alt.EncodingSortField(field='count', op='sum', order='descending')),

    color=alt.condition(

        alt.datum.title=='Data Engineer',

        alt.value('#b00200'),

        alt.value('silver')

    )

    

)



text = alt.Chart(data).mark_text(

    align='left',

    baseline='middle',

    dx=5

).encode(

    text='count:Q', 

    x=alt.X('count:Q', axis=None), 

    y=alt.Y('title:N', sort=alt.EncodingSortField(field='count', op='sum', order='descending')),

    color=alt.condition(

        alt.datum.title=='Data Engineer',

        alt.value('#b00200'),

        alt.value('#FFFFFF')

    )

)



(bars+text).properties(

    title='Data Engineers make up a large number of the least respondants',

    width=600,

    height=400

).configure_title(

    dy=-10,

    fontSize=20,

    anchor='start',

    color='#b00200'

).configure_axis(

    grid=False

).configure_view(

    strokeOpacity=0

)

de = df[df['Q5']=='Data Engineer']

data = pd.DataFrame({'age': de.Q1})



base = alt.Chart(data).transform_calculate(

    age_min='parseInt(split(datum.age, "-")[0])',

    age_max='parseInt(split(datum.age, "-")[1]) + 1',

    age_mid='(datum.age_min + datum.age_max) / 2'

)



bars = base.mark_bar(color='silver').encode(

    x=alt.X('age_min:Q', title='Age', bin='binned', scale=alt.Scale(domain=(18, 75))),

    x2='age_max:Q',

    y=alt.Y('count():Q', title='Responses')

)



median = base.mark_rule(size=5, color='#b00200').encode(

    x = 'median(age_mid):Q',

)



(bars+median).properties(

    title='Most Data Engineers are between the ages of 18 and 70+',

    width=600,

    height=400

).configure_title(

    dy=-10,

    fontSize=20,

    anchor='start',

    color='#b00200'

).configure_axis(

    grid=False

).configure_view(

    strokeOpacity=0

)
data = pd.DataFrame({'income': de.Q10})

data = data.dropna()

order = pd.DataFrame({'income':data.income.dropna().unique()})

order['sort'] = order.income.apply(lambda x: int(re.sub('[$>,]', '', x.split('-')[0])))

order = order.sort_values('sort', ascending=False).reset_index(drop=True)





bars = alt.Chart(data).mark_bar().encode(

    y=alt.Y('income:N', title='Income', sort=list(order.income)),

    x=alt.X('count():Q', title='', axis=None),

    color=alt.condition(

        alt.datum.income=='> $500,000',

        alt.value('#b00200'),

        alt.value('silver')

    )

)



text = alt.Chart(data).mark_text(

    align='center',

    baseline='middle',

    dx=10

).encode(

    text='count():Q', 

    y=alt.Y('income:O', title='', sort=list(order.income)),

    x=alt.X('count():Q', title='', axis=None),

    color=alt.condition(

        alt.datum.income=='> $500,000',

        alt.value('#b00200'),

        alt.value('#FFFFFF')

    )

)





(bars+text).properties(

    title='I want to be friends with these Data Engineers',

    width=400,

    height=600

).configure_title(

    dy=-10,

    fontSize=20,

    anchor='start',

    color='#b00200'

).configure_axis(

    grid=False

).configure_view(

    strokeOpacity=0

)
