import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
df = pd.read_csv("/kaggle/input/valorant-weapon-stats/valorant-stats.csv")
df.head()
weapon_type = dict(Counter(df['Weapon Type']))
weapon_type = {'Weapon Type': list(weapon_type.keys()), 'count': list(weapon_type.values())}

fig_weapon = px.pie(weapon_type, values = 'count', names = 'Weapon Type', title = 'Weapon Type Distribution', hole = .5, color_discrete_sequence = px.colors.sequential.Agsunset)
fig_weapon.show()
def return_sorted(col_name, asc = False, limit = 5):
    sorted_df = df.sort_values(by=col_name, ascending=asc)
    sorted_df = sorted_df[:limit]

    return {'weapon': sorted_df['Name'].to_list(), 'values': sorted_df[col_name].to_list()}
headshot_dict = return_sorted('HDMG_0', limit = 10)

num_ele = len(headshot_dict['weapon'])
colors = ['#22a6b3',] * num_ele
colors[0] = '#eb4d4b'

fig_headshot = go.Figure(data=[go.Bar(
    x=headshot_dict['weapon'],
    y=headshot_dict['values'],
    marker_color = colors
)])

fig_headshot.update_traces(texttemplate='%{y:}', textposition='outside')

fig_headshot.update_layout(
    title = 'Damage given by a headshot',
    yaxis=dict(
        title='Damage',
        titlefont_size=16,
        tickfont_size=14,
    ),
    width=800,
    height=800
)

fig_headshot.show()
mag_dict = return_sorted('Magazine Capacity', limit = 10)

num_ele = len(mag_dict['weapon'])
colors = ['#6c5ce7',] * num_ele

fig_mag = go.Figure(data=[go.Bar(
    x=mag_dict['weapon'],
    y=mag_dict['values'],
    marker_color = colors
)])

fig_mag.update_traces(texttemplate='%{y:}', textposition='outside')

fig_mag.update_layout(
    yaxis=dict(
        title='Magazine Capacity',
        titlefont_size=16,
        tickfont_size=14,
    ),
    width=800,
    height=800
)

fig_mag.show()
pen_type = dict(Counter(df['Wall Penetration']))
pen_type = {'Penetration': list(pen_type.keys()), 'count': list(pen_type.values())}

colors_pie = ['#fed330', '#20bf6b', '#eb3b5a']

fig_pen = px.pie(pen_type, values = 'count', names = 'Penetration', title = 'Penetration Distribution', hole = .5, color_discrete_sequence = colors_pie)
fig_pen.show()