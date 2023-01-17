import altair as alt
from altair import datum
import numpy as np
import pandas as pd

alt.renderers.enable('notebook')
performance = pd.read_csv('../input/StudentsPerformance.csv')
performance.head()
performance.describe(include='all')
performance.shape
brush = alt.selection(type='interval')
points = alt.Chart().mark_point().encode(
    x='lunch',
    y='math score'
).properties(
    width=400
).add_selection(
    brush
)

bars = alt.Chart().mark_bar().encode(
    x = 'parental level of education',
    y = 'count()'
).transform_filter(
    brush
).properties(
    width=400
)

alt.hconcat(points, bars, data=performance)
alt.Chart(performance).mark_bar().encode(
    x = 'gender',
    y = 'count()'
).properties(
    width = 400,
    title = "Count of Gender"
)
alt.Chart(performance).mark_bar().encode(
    x = 'race/ethnicity',
    y = 'count()'
).properties(
    width = 400,
    title = "Race/Ethnicity and its count"
)
performance.loc[(performance['parental level of education'] == 'high school') |
                (performance['parental level of education'] == 'some high school'), ['parental level of education']] = 'school'

performance.loc[(performance['parental level of education'] == "associate's degree") |
                (performance['parental level of education'] == "bachelor's degree") | 
                (performance['parental level of education'] == "master's degree"), ['parental level of education']] = 'professional_degree'
brush = alt.selection_multi(encodings=['x'])
points = alt.Chart().mark_bar().encode(
    x='test preparation course',
    y='count()',
    color = alt.condition(brush,
                      alt.Color('test preparation course:N', legend=None),
                      alt.value('lightgray'))
).properties(
    width=400
).add_selection(
    brush
)

bars = alt.Chart().mark_bar().encode(
    y = 'count()'
).transform_filter(
    brush
).properties(
    width=400
)

#alt.hconcat(points, bars, data=performance)

alt.vconcat(alt.hconcat(points, bars.encode(x = 'parental level of education:N'), data = performance),
alt.hconcat(bars.encode(x = 'lunch:N'), bars.encode(x = 'race/ethnicity:N'), data = performance), bars.encode(x = 'gender:N'), data = performance)
alt.Chart(performance).mark_circle().encode(
    x='race/ethnicity:N',
    y='parental level of education:N',
    size='count():Q',
    tooltip=['count()']
).properties(
    width = 400
)
line = alt.Chart(performance).mark_line(point=True).encode(
    y='count()'
)

alt.hconcat(line.encode(x = 'math score'), line.encode(x = 'reading score'), line.encode(x = 'writing score'))
bar = alt.Chart(performance).mark_bar().encode(
    y = 'count()'
).properties(
    width = 300
).transform_filter(
    (datum['math score'] <= 40) &\
    (datum['reading score'] <= 40) &\
    (datum['writing score'] <= 40)
)

(bar.encode(x = 'gender') |  bar.encode(x = 'lunch')) & \
(bar.encode(x = 'race/ethnicity') | bar.encode(x = 'parental level of education')) & \
bar.encode(x = 'test preparation course')
bar = alt.Chart(performance).mark_bar().encode(
    y = 'count()'
).properties(
    width = 300
).transform_filter(
    (datum['math score'] <= 40) |\
    (datum['reading score'] <= 40) |\
    (datum['writing score'] <= 40) & 
    ~((datum['math score'] <= 40) &\
    (datum['reading score'] <= 40) &\
    (datum['writing score'] <= 40))
)

(bar.encode(x = 'gender') |  bar.encode(x = 'lunch')) & \
(bar.encode(x = 'race/ethnicity') | bar.encode(x = 'parental level of education')) & \
bar.encode(x = 'test preparation course')
bar = alt.Chart(performance).mark_bar().encode(
    y = 'count()'
).properties(
    width = 300
).transform_filter(
    (datum['math score'] >= 80) &\
    (datum['reading score'] >= 80) &\
    (datum['writing score'] >= 80)
)

(bar.encode(x = 'gender') |  bar.encode(x = 'lunch')) & \
(bar.encode(x = 'race/ethnicity') | bar.encode(x = 'parental level of education')) & \
bar.encode(x = 'test preparation course')
base = alt.Chart(performance)

area_args = {'opacity': .3, 'interpolate': 'step'}
blank_axis = alt.Axis(title='')

points = base.mark_circle().encode(
    color='parental level of education'
)

points.encode(x = 'reading score', y = 'writing score') &\
points.encode(x = 'math score', y = 'writing score') &\
points.encode(x = 'math score', y = 'reading score')


