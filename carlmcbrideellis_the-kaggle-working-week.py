import pandas as pd

import plotly.express as px

diasDeLaSemana = [ 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

# read in the data

competitions_df = pd.read_csv('../input/meta-kaggle/Submissions.csv')

# add a column of 1's to sum up

competitions_df['ones'] = 1

# convert the date/times

competitions_df['SubmissionDate'] = pd.to_datetime(competitions_df['SubmissionDate'])

# sum things up by the day of the week

week_df = competitions_df.groupby(competitions_df['SubmissionDate'].dt.day_name()).sum().reindex(diasDeLaSemana)
# and now make a plot

px.bar(week_df, y=['ones'], color=week_df['ones'].values , title="Competition submissions").show()
# read in the data

datasets_df = pd.read_csv('../input/meta-kaggle/Datasets.csv')

# add a column to sum up

datasets_df['ones'] = 1

# convert the date/times

datasets_df['CreationDate'] = pd.to_datetime(datasets_df['CreationDate'])

# sum things up by the day of the week

week_df = datasets_df.groupby(datasets_df['CreationDate'].dt.day_name()).sum().reindex(diasDeLaSemana)
px.bar(week_df, y=['ones'], color=week_df['ones'].values , title="Dataset creation").show()
# read in the data

notebooks_df = pd.read_csv('../input/meta-kaggle/Kernels.csv')

# add a column to sum up

notebooks_df['ones'] = 1

# convert the date/times

notebooks_df['CreationDate'] = pd.to_datetime(notebooks_df['CreationDate'])

# sum things up by the day of the week

week_df = notebooks_df.groupby(notebooks_df['CreationDate'].dt.day_name()).sum().reindex(diasDeLaSemana)
px.bar(week_df, y=['ones'], color=week_df['ones'].values , title="Notebooks creation").show()
# read in the data

topics_df = pd.read_csv('../input/meta-kaggle/ForumTopics.csv')

# add a column to sum up

topics_df['ones'] = 1

# convert the date/times

topics_df['CreationDate'] = pd.to_datetime(topics_df['CreationDate'])

# sum things up by the day of the week

week_df = topics_df.groupby(topics_df['CreationDate'].dt.day_name()).sum().reindex(diasDeLaSemana)
px.bar(week_df, y=['ones'], color=week_df['ones'].values , title="Discussions: Topic posts").show()
# read in the data

messages_df = pd.read_csv('../input/meta-kaggle/ForumMessages.csv')

# add a column to sum up

messages_df['ones'] = 1

# convert the date/times

messages_df['PostDate'] = pd.to_datetime(messages_df['PostDate'])

# sum things up by the day of the week

week_df = messages_df.groupby(messages_df['PostDate'].dt.day_name()).sum().reindex(diasDeLaSemana)
px.bar(week_df, y=['ones'], color=week_df['ones'].values , title="Discussions: Replies").show()