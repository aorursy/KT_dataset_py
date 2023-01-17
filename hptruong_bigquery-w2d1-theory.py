# Welcome back Mr. Truong!

# Well, actually you were late for one week. Actually, we know that you've neglecting Pick A Kit for two weeks...

# Just curious, are you giving up on Pick A Kit by any chance?

# How do you know so much about me, man? Darn, I'm scared the hell out of Google now.

# Nope, we are not Google. This is your deeper part inside you talking to yourself.

# F... Whatever, man!

# Yeah, I'm giving up on it because... let's put it straight, the filter project for example, I feel that if I don't do anything,

# then no one cares to do anything. Since I feel that I'm being exploited, why the hell I have to spend my time on something that rewards me

# almost nothing? And by rewards I mean, profit, teamwork, work satisfaction, etc. I know Bob is busy, but com'on man, I don't want to torture

# myself. Unless Bob changes his mind/his working habits to be more serious, it's just hard for me.

# Anyways, let's get back to the lesson.

# ... (moment of silence). Yes, sir. Today we're to talk about Order By, which you have known so well.

# First, why don't you show us how much knowledge still remains after your vacation? Let's start by showing the first 5 rows of the

# accident_2015 table, which contains information on traffic accidents in the US where at least one person died



# call BigQuery

from google.cloud import bigquery

# create an account

client = bigquery.Client()

# create dataset handler

dataset_ref = client.dataset("nhtsa_traffic_fatalities", project="bigquery-public-data")

# fetch the dataset

dataset = client.get_dataset(dataset_ref)

# create table handler

table_ref = dataset_ref.table('accident_2015')

# fetch the table

table = client.get_table(table_ref)



# show first five rows

client.list_rows(table, max_results=5).to_dataframe()
# Hmmm. I saw you were quite of shaky, Mr. Truong. Did you also sweat? Forgot most of the stuff already

# (stunt silence for 2 secs). Hey man, I did take my notes!

# Alright, alright. Just checkin'. Now the column called `consecutive_number` is the unique ID of each acc.

# The column `timestamp_of_crash` is the date and time of the accident.

# Can you count how many accidents within the day of the week?

# Ok, let me try



# First, define the query that do this job

query = """

        SELECT EXTRACT(DAYOFWEEK from timestamp_of_crash) AS day_of_week

              ,COUNT(consecutive_number) AS num_of_accidents

        FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`

        GROUP BY day_of_week

        ORDER BY num_of_accidents DESC

        """
# Set up the query. Cancel it if more than 1GB

safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**9)

safe_query_job = client.query(query, job_config=safe_config)

safe_query_job.to_dataframe()
# Sigh... Shaky again, right?

# Your notes are great, but not you. You're not as smart as you think, so don't go out and make fun of 

# other people, especially, Bob. Just, shut up and do your work instead of complaining about Bob. He's

# busy with his stuff too.

# Hey you are so mean to me, BigQuery. At least I always take notes, while I know for a fact that 

# Bob rarely take notes and the only time he does is to complement someone.

# So what are you suggesting?

# Well, I'm saying that being too nice hurts you in the long run.

# How is that related to the lesson?

# No, bursting out feelings and negative thoughts, is that OK?

# Be cool man, I'm just saying. Let me quickly jump back to the lesson.

# If you look at BigQuery documentation, 1 is referred to as Sunday while 7 is referred to as Saturday.

# So congratulations, you've confirmed the fact that Saturday is the dangerous day to drive.
# Am I done?

# Nope, head to the exercise!