import pandas as pd
import plotly.express as px
from collections import Counter
df = pd.read_csv("/kaggle/input/pubg-weapon-stats/pubg-weapon-stats.csv")
df.head()
bullet_counts = dict(Counter(df['Bullet Type']))

# Some preprocessing to display values in the right format
bullet_types = []
for bt in list(bullet_counts.keys()):
    if bt in (7.62, 5.56, 9):
        bullet_types.append(str(bt) + " mm")
    elif bt == .45:
        bullet_types.append(".45 ACP")
    elif bt == .3:
        bullet_types.append(".300 Magnum")
    else:
        bullet_types.append("Unknown")

bullet_counts = {'Bullet Type': bullet_types, 'count': list(bullet_counts.values())}

fig_bullet = px.pie(bullet_counts, values = 'count', names = 'Bullet Type', title = 'Bullet Type', hole = .5, color_discrete_sequence = px.colors.diverging.Portland)
fig_bullet.show()
fire_type = dict(Counter(df['Fire Mode']))
fire_type = {'Fire Type': list(fire_type.keys()), 'count': list(fire_type.values())}

fig_fire = px.pie(fire_type, values = 'count', names = 'Fire Type', title = 'Fire Type', hole = .5, color_discrete_sequence = px.colors.sequential.Agsunset)
fig_fire.show()
def return_sorted(col_name, asc = False, limit = 5):
    sorted_df = df.sort_values(by=col_name, ascending=asc)
    sorted_df = sorted_df[:limit]

    return {'weapon': sorted_df['Weapon Name'].to_list(), 'values': sorted_df[col_name].to_list()}
mag_dict = return_sorted('Magazine Capacity')

import plotly.express as px
import plotly.graph_objects as go

num_ele = len(mag_dict['weapon'])
colors = ['#22a6b3',] * num_ele

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
lethal_dict = return_sorted('Damage', limit = 10)

num_ele = len(lethal_dict['weapon'])
colors = ['#eb4d4b',] * num_ele

fig_lethal = go.Figure(data=[go.Bar(
    x=lethal_dict['weapon'],
    y=lethal_dict['values'],
    marker_color = colors
)])

fig_lethal.update_traces(texttemplate='%{y:}', textposition='outside')

fig_lethal.update_layout(
    title = 'Most lethal weapons',
    yaxis=dict(
        title='Damage',
        titlefont_size=16,
        tickfont_size=14,
    ),
    width=800,
    height=800
)

fig_lethal.show()
headshot_dict = return_sorted('Shots to Kill (Head)', asc = True, limit = 10)

num_ele = len(headshot_dict['weapon'])
colors = ['#6ab04c',] * num_ele

fig_headshot = go.Figure(data=[go.Bar(
    x=headshot_dict['weapon'],
    y=headshot_dict['values'],
    marker_color = colors
)])

fig_headshot.update_traces(texttemplate='%{y:}', textposition='outside')

fig_headshot.update_layout(
    title = 'Number of bullets needed by a weapon to kill by headshot',
    yaxis=dict(
        title='Bullets Needed',
        titlefont_size=16,
        tickfont_size=14,
    ),
    width=800,
    height=800
)

fig_headshot.show()
range_dict = return_sorted('Range', asc = False, limit = 10)

num_ele = len(range_dict['weapon'])
colors = ['#30336b',] * num_ele

fig_range = go.Figure(data=[go.Bar(
    x=range_dict['weapon'],
    y=range_dict['values'],
    marker_color = colors
)])

fig_range.update_traces(texttemplate='%{y:}', textposition='outside')

fig_range.update_layout(
    title = 'Long-Range Weapons',
    yaxis=dict(
        title='Range (in m)',
        titlefont_size=16,
        tickfont_size=14,
    ),
    width=800,
    height=800
)

fig_range.show()
weapon_counts = dict(Counter(df['Weapon Type']))

weapon_counts = {'Weapon Type': list(weapon_counts.keys()), 'count': list(weapon_counts.values())}

fig_weapons = px.pie(weapon_counts, values = 'count', names = 'Weapon Type', title = 'Weapon Type Distribution', hole = 0.3, color_discrete_sequence = px.colors.diverging.Temps)
fig_weapons.show()