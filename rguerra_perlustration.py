from pprint import pformat, pprint
#Parameters
min_post_length = 10
max_post_length = 100
num_posts = 10000
sql_conn = sqlite3.connect('../input/database.sqlite')

# Loading the data into a Panda's Data Frame
df = pd.read_sql(
      "SELECT score, subreddit, body, created_utc FROM May2015 ORDER BY RANDOM()"+\
      "LIMIT {}".format(num_posts), 
      sql_conn)

pprint("Data shape: {}".format(df.shape))
pprint("Headers: {}".format(df.columns.values))
pprint(df[:5])
import dateutil.parser
import datetime
a_post = df[:1]
created_utc = a_post['created_utc'][0]
print(created_utc)

# The following code will parse the creating times.
status_published = datetime.datetime.fromtimestamp(created_utc)
print(status_published)

# Transform the UTC Timestamp into a Date
df['Date'] = df['created_utc'].apply(lambda x: datetime.datetime.fromtimestamp(x))
print(df['Date'])