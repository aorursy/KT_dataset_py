# Import library

import sqlite3

import pandas as pd



# Create connection with SQLite database

conn = sqlite3.connect(":memory:")



# Create dataframes by importing reviews.csv and pricing_plans.csv, and transform them into sqlite file.

df = pd.read_csv('../input/reviews.csv')

df.to_sql('reviews', conn, if_exists='append', index=False)

print('ok')

df = pd.read_csv('../input/pricing_plans.csv')

df.to_sql('pricing_plans', conn, if_exists='append', index=False)

print('ok')
# Check the head of two dataframes

pd.read_sql("""



SELECT *

FROM reviews

LIMIT 50



""",con=conn)



pd.read_sql("""



SELECT *

FROM pricing_plans

LIMIT 50



""",con=conn)
# Check the count of different rating levels

pd.read_sql("""



SELECT rating,count(*) as [count]

FROM reviews

GROUP BY rating



""",con = conn)
# Check the count of different rating levels

pd.read_sql("""



SELECT rating,count(*) as [count]

FROM reviews

GROUP BY rating



""",con = conn)
# Join pricing,ratings and helpful_count by app_url.

pd.read_sql("""



    SELECT rating, helpful_count, price

    FROM reviews

    JOIN pricing_plans

    ON reviews.app_url = pricing_plans.app_url



""",con = conn)
# Check the ratings of apps with free charging.

pd.read_sql("""



SELECT *

FROM (

    SELECT rating, helpful_count, price

    FROM reviews

    JOIN pricing_plans

    ON reviews.app_url = pricing_plans.app_url

    )

WHERE price = 'Free' OR price = 'Free to install'

ORDER BY rating DESC



""",con=conn)



# We also want to know the count of different rating levels among free apps.

pd.read_sql("""



SELECT rating,count(*) as [count]

FROM (

    SELECT rating, helpful_count, price

    FROM reviews

    JOIN pricing_plans

    ON reviews.app_url = pricing_plans.app_url

    )

WHERE price = 'Free' OR price = 'Free to install'

GROUP BY rating

ORDER BY rating DESC



""",con=conn)

# We can see that most free apps are rated 5.
# So what about ratings of charged apps?

pd.read_sql("""



SELECT *

FROM (

    SELECT rating, helpful_count, price

    FROM reviews

    JOIN pricing_plans

    ON reviews.app_url = pricing_plans.app_url

    )

WHERE price != 'Free' AND price != 'Free to install'

ORDER BY rating DESC



""",con=conn)



# and the count of their rating levels

pd.read_sql("""



SELECT rating,count(*) as [count]

FROM (

    SELECT rating, helpful_count, price

    FROM reviews

    JOIN pricing_plans

    ON reviews.app_url = pricing_plans.app_url

    )

WHERE price != 'Free' AND price != 'Free to install'

GROUP BY rating

ORDER BY rating DESC



""",con=conn)

#Oops, it seems that most charged apps are also rated 5.

# How many apps rated 5 in total?

pd.read_sql("""



SELECT COUNT(*)

FROM (

    SELECT rating, helpful_count, price

    FROM reviews

    JOIN pricing_plans

    ON reviews.app_url = pricing_plans.app_url

    )

WHERE rating = 5



""",con=conn)

# Among all the apps rated 5, how many are free apps?

pd.read_sql("""



SELECT price,count(*) as [count]

FROM (

    SELECT rating, helpful_count, price

    FROM reviews

    JOIN pricing_plans

    ON reviews.app_url = pricing_plans.app_url

    )

WHERE (price = 'Free' OR price = 'Free to install') AND rating = 5



""",con=conn)
# Then what about the helpful_count of free apps?

pd.read_sql("""



SELECT helpful_count,count(*) as [count]

FROM (

    SELECT rating, helpful_count, price

    FROM reviews

    JOIN pricing_plans

    ON reviews.app_url = pricing_plans.app_url

    )

WHERE price = 'Free' OR price = 'Free to install'

GROUP BY helpful_count

ORDER BY helpful_count DESC



""",con=conn)



# as well as the helpful_count of charged apps

pd.read_sql("""



SELECT helpful_count,count(*) as [count]

FROM (

    SELECT rating, helpful_count, price

    FROM reviews

    JOIN pricing_plans

    ON reviews.app_url = pricing_plans.app_url

    )

WHERE price != 'Free' AND price != 'Free to install'

GROUP BY helpful_count

ORDER BY helpful_count DESC



""",con=conn)