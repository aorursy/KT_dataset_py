import pandas as pd

import glob



# For plots we use plotly express and

import plotly.express as px

import plotly.graph_objects as go

from plotly.subplots import make_subplots

from plotly.colors import n_colors
train_df = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/train.csv")

print(train_df.head())

print(train_df.shape)
print(len(train_df['Patient'].unique()))

#print(train_df['Patient'].unique())
print(train_df['Weeks'].describe())

# print(train_df[train_df['Weeks'].isnull()])
value_count_week = train_df['Weeks'].value_counts()





data  = go.Bar(

            x = value_count_week.index,

            y = value_count_week.values,

            width = 1.0,

            marker_color='orangered')

layout = go.Layout(

            height = 500,

            width = 1200,

            xaxis=dict(range=[-5, 130], autorange=False, zeroline=False, title="Appointment week"),

            yaxis=dict(zeroline=False, title='Number of patients on week'),

            title = "Which weeks are people most coming in for checkup",

        )

fig  = go.Figure(data=data, layout=layout)

fig.show()
from plotly.subplots import make_subplots

fig1 = px.scatter(train_df, x="Weeks", y="Patient", color='SmokingStatus',

                 title="Appointment distribution for all patients",

                 labels={"Weeks":"Appointment week (when they showed up)"})

fig1.update_yaxes(showticklabels=False)

fig1.show()
total_appointment = train_df.groupby(['Patient', 'SmokingStatus']).size().reset_index(name="total_appointment")

total_appointment_20wks = train_df[train_df["Weeks"] <= 20].groupby(['Patient']).size().reset_index(name="appt_within_20_Weeks")

total_appointment_40wks = train_df[train_df["Weeks"] <= 40].groupby(['Patient']).size().reset_index(name="appt_within_40_Weeks")



fig = make_subplots(rows=1, cols=3,

                   subplot_titles=("Total Appointment within 20 weeks",

                                   "Total Appointment within weeks",

                                   "Total Appointment count all weeks"))

fig.add_trace(go.Scatter(

    x=total_appointment_20wks['Patient'], y=total_appointment_20wks['appt_within_20_Weeks'],

    mode='markers'

), row=1, col=1)

fig.add_trace(go.Scatter(

    x=total_appointment_40wks['Patient'], y=total_appointment_40wks['appt_within_40_Weeks'],

    mode='markers'

), row=1, col=2)

fig.add_trace(go.Scatter(

    x=total_appointment['Patient'], y=total_appointment['total_appointment'],

    mode='markers'

), row=1, col=3)

fig.update_xaxes(showticklabels=False, title_text='Patient')

fig.update_xaxes(title_text='Count appointments')

fig.update_layout(showlegend=False, title_text="Comparing appointments within various week intervals")

fig.show()
total_appontment_20wks = train_df[train_df["Weeks"] <= 20].groupby(['Patient', 'SmokingStatus']).size().reset_index(name="appt_within_20_Weeks")

fig = px.scatter(total_appontment_20wks, x="Patient", y="appt_within_20_Weeks",

                 facet_col='SmokingStatus', color_discrete_sequence=["coral"])

fig.update_xaxes(showticklabels=False, title_text='Patient')

fig.update_yaxes(title_text='Count appointments')

fig.show()



total_appontment_40wks = train_df[train_df["Weeks"] <= 40].groupby(['Patient', 'SmokingStatus']).size().reset_index(name="appt_within_40_Weeks")

fig = px.scatter(total_appontment_40wks, x="Patient",

                 y="appt_within_40_Weeks", facet_col='SmokingStatus',

                 color_discrete_sequence=["goldenrod"])

fig.update_xaxes(showticklabels=False, title_text='Patient')

fig.update_yaxes(title_text='Count appointments')

fig.show()



fig = px.scatter(total_appointment, x="Patient",

                 y="total_appointment", facet_col='SmokingStatus',

                 color_discrete_sequence=["blueviolet"])

fig.update_xaxes(showticklabels=False, title_text='Patient')

fig.update_yaxes(title_text='Count appointments')

fig.show()
fig = px.scatter(train_df, x="Weeks", y="FVC", color='SmokingStatus',

                 title="Appointment distribution based on FCV level",

                 labels={"Weeks":"Appointment week (when they showed up)"})



fig.show()
smoking_status = train_df[train_df['SmokingStatus']=="Ex-smoker"]

fig = go.Figure()



colors = n_colors('rgb(5, 200, 200)', 'rgb(200, 10, 10)', len(train_df['Patient'].unique()), colortype='rgb')



for (j, color) in zip(train_df['Patient'].unique(), colors):

    data = smoking_status[smoking_status['Patient'] == j]

    fig.add_trace(go.Violin(

        x = data['FVC'], line_color=color

    ))

fig.update_layout(

    showlegend=False, title_text="FVC distribution - Ex smokers"

)

fig.update_yaxes(showticklabels=False, title_text='Patient')

fig.update_traces(orientation='h', side='positive', width=12, points=False)

fig.update_layout(xaxis_showgrid=False, xaxis_zeroline=False)



fig.show()







smoking_status = train_df[train_df['SmokingStatus']=="Never smoked"]

fig = go.Figure()

for (j, color) in zip(train_df['Patient'].unique(), colors):

    data = smoking_status[smoking_status['Patient'] == j]

    fig.add_trace(go.Violin(

        x = data['FVC'], line_color=color

    ))

fig.update_layout(

    showlegend=False, title_text="FVC distribution - Never smoked patients"

)

fig.update_yaxes(showticklabels=False, title_text='Patient')

fig.update_traces(orientation='h', side='positive', width=8, points=False)

fig.update_layout(xaxis_showgrid=False, xaxis_zeroline=False)

fig.show()





smoking_status = train_df[train_df['SmokingStatus']=="Currently smokes"]

fig = go.Figure()

for (j, color) in zip(train_df['Patient'].unique(), colors):

    data = smoking_status[smoking_status['Patient'] == j]

    fig.add_trace(go.Violin(

        x = data['FVC'], line_color=color

    ))

fig.update_layout(

    showlegend=False, title_text="FVC distribution - Currently smokes"

)

fig.update_yaxes(showticklabels=False, title_text='Patient')

fig.update_traces(orientation='h', side='positive', width=8, points=False)

fig.update_layout(xaxis_showgrid=False, xaxis_zeroline=False)

fig.show()
fig = px.violin(train_df, y="FVC", x="SmokingStatus", box=True, # draw box plot inside the violin

                points='all', # can be 'outliers', or False

               )

fig.show()
smoking_status = train_df[train_df['SmokingStatus']=="Ex-smoker"]

fig = go.Figure()



colors = n_colors('rgb(5, 200, 200)', 'rgb(200, 10, 10)', len(train_df['Patient'].unique()), colortype='rgb')



for (j, color) in zip(train_df['Patient'].unique(), colors):

    data = smoking_status[smoking_status['Patient'] == j]

    fig.add_trace(go.Violin(

        x = data['Percent'], line_color=color

    ))

fig.update_layout(

    showlegend=False, title_text="Percent distribution - Ex smokers"

)

fig.update_yaxes(showticklabels=False, title_text='Patient')

fig.update_traces(orientation='h', side='positive', width=12, points=False)

fig.update_layout(xaxis_showgrid=False, xaxis_zeroline=False)



fig.show()







smoking_status = train_df[train_df['SmokingStatus']=="Never smoked"]

fig = go.Figure()

for (j, color) in zip(train_df['Patient'].unique(), colors):

    data = smoking_status[smoking_status['Patient'] == j]

    fig.add_trace(go.Violin(

        x = data['Percent'], line_color=color

    ))

fig.update_layout(

    showlegend=False, title_text="Percent distribution - Never smoked patients"

)

fig.update_yaxes(showticklabels=False, title_text='Patient')

fig.update_traces(orientation='h', side='positive', width=8, points=False)

fig.update_layout(xaxis_showgrid=False, xaxis_zeroline=False)

fig.show()





smoking_status = train_df[train_df['SmokingStatus']=="Currently smokes"]

fig = go.Figure()

for (j, color) in zip(train_df['Patient'].unique(), colors):

    data = smoking_status[smoking_status['Patient'] == j]

    fig.add_trace(go.Violin(

        x = data['Percent'], line_color=color

    ))

fig.update_layout(

    showlegend=False, title_text="Percent distribution - Currently smokes"

)

fig.update_yaxes(showticklabels=False, title_text='Patient')

fig.update_traces(orientation='h', side='positive', width=8, points=False)

fig.update_layout(xaxis_showgrid=False, xaxis_zeroline=False)

fig.show()
statistical_df = train_df.groupby(['Patient', 'SmokingStatus', 'Age', 'Sex']).agg(

            {'FVC': ['mean', 'median', 'min', 'max']})

statistical_df.columns = ['fvc_mean', 'fvc_median', 'fvc_min', 'fvc_max']

statistical_df = statistical_df.reset_index()

statistical_df.head()
fig = px.box(statistical_df, x="Sex", y="fvc_mean", color="SmokingStatus")

fig.update_layout(

    showlegend=True, title_text="Mean FVC distribution among smokers"

)

fig.show()



fig = px.box(statistical_df, x="Sex", y="fvc_median", color="SmokingStatus")

fig.update_layout(

    showlegend=True, title_text="Median FVC distribution among smokers"

)

fig.show()



fig = px.box(statistical_df, x="Sex", y="fvc_min", color="SmokingStatus")

fig.update_layout(

    showlegend=True, title_text="Minimum FVC distribution among smokers"

)

fig.show()

fig = px.box(statistical_df, x="Sex", y="fvc_max", color="SmokingStatus")

fig.update_layout(

    showlegend=True, title_text="Maximum FVC distribution among smokers"

)

fig.show()
fig = px.violin(statistical_df, x='Sex', y="Age", color="SmokingStatus", box=True, points='all')

fig.update_layout(

    showlegend=True, title_text="Age distribution"

)

fig.show()
fig = px.scatter(statistical_df, x='fvc_mean', y='Age', color='Sex', facet_col='SmokingStatus', facet_col_wrap=4)

fig.show()
train_df[train_df['Patient'] == "ID00336637202286801879145"]['Weeks'].diff()

# We got to find the time lapsed difference for each of the persons
train_df[train_df['Patient'] == 'ID00007637202177411956430'].shape

#Something to keep in mind, You need to validate that the number of test recorded is the same and the numbe of dcm file provided for each of the Patient ID
# smoking_status = train_df[train_df['SmokingStatus']=="Currently smokes"]

# fig = go.Figure()

# for j in train_df['Patient'].unique():

#     data = smoking_status[smoking_status['Patient'] == j]

#     fig.add_trace(go.Violin(

#         x = data['Weeks'], y=data['FVC']

#     ))

# fig.update_layout(

#     showlegend=False, title_text="FVC distribution - Currently smokes"

# )



# fig.show()





# for j in train_df['Patient'].unique():

#     data = smoking_status[smoking_status['Patient'] == j]

#     fig.add_trace(go.Scatter(

#         x = data['Weeks'], y=data['FVC']

#     ))

# fig.update_layout(

#     showlegend=False, title_text="FVC distribution - Ex smokers"

# )



# fig.show()