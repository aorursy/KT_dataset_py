# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Set your own project id here

PROJECT_ID = 'kaggle-playground-170215'

from google.cloud import bigquery

bigquery_client = bigquery.Client(project=PROJECT_ID)
query = """

select * from `kaggle-playground-170215.metakaggle_interview.Datasets` """

query_job = bigquery_client.query(query)  # Make an API request.



query_job.result().to_dataframe()
query = """

SELECT COUNT(Id) as new_users, RegisterDate 

FROM `kaggle-playground-170215.metakaggle_interview.Users`

WHERE RegisterDate < CURRENT_DATE()

    AND RegisterDate > DATE_SUB(CURRENT_DATE, INTERVAL 60 DAY)

GROUP BY RegisterDate

ORDER BY RegisterDate DESC"""

query_job = bigquery_client.query(query)  # Make an API request.

query_job.result().to_dataframe()
query = """

            SELECT

              table1.id,

              table1.UserName,

              table1.RegisterDate,

              IF(table1.RegisterDate = table2.FirstJoinDate, TRUE, FALSE) as JoinedOrg,

              IF(table1.RegisterDate = table2.FirstFollowDate, TRUE, FALSE) as Followed,

              IF(table1.RegisterDate = table2.FirstKernelVoteDate, TRUE, FALSE) as KernelVoted,

              IF(table1.RegisterDate = table2.FirstMsgVoteDate, TRUE, FALSE) as MsgVoted,

              IF(table1.RegisterDate = DATE(table2.FirstPostDate), TRUE, FALSE) as MessagePosted

            FROM 

                `kaggle-playground-170215.metakaggle_interview.Users` as table1

                LEFT JOIN

                (

                    SELECT t2.UserId, t2.FirstJoinDate, t3.FirstFollowDate, t4.FirstKernelVoteDate, t5.FirstMsgVoteDate, t6.FirstPostDate

                    FROM (

                        (select distinct UserId, min(JoinDate) as FirstJoinDate from `kaggle-playground-170215.metakaggle_interview.UserOrganizations`

                                group by UserId) as t2

                        JOIN 

                            (select distinct UserId, min(CreationDate) as FirstFollowDate from `kaggle-playground-170215.metakaggle_interview.UserFollowers`

                                group by UserId) as t3

                            ON t2.UserId = t3.UserId 

                        JOIN 

                            (select distinct UserId, min(VoteDate) as FirstKernelVoteDate from `kaggle-playground-170215.metakaggle_interview.KernelVotes`

                                group by UserId) as t4

                            ON t2.UserId = t4.UserId

                        JOIN 

                            (select distinct FromUserId, min(VoteDate) as FirstMsgVoteDate from `kaggle-playground-170215.metakaggle_interview.ForumMessageVotes`

                                group by FromUserId) as t5

                            ON t2.UserId = t5.FromUserId

                        JOIN 

                            (select distinct PostUserId, min(PostDate) as FirstPostDate from `kaggle-playground-170215.metakaggle_interview.ForumMessages`

                                group by PostUserId) as t6

                            ON t2.UserId = t6.PostUserId

                    )

                ) as table2

                ON table1.Id = table2.UserId

            LIMIT 100

        """

query_job = bigquery_client.query(query)  # Make an API request.

query_job.result().to_dataframe()

query = """

SELECT 

    HostSegmentTitle,

    FORMAT_DATE("%Y", DATE(EnabledDate)) AS Year,

    1 AS amount 



FROM `kaggle-playground-170215.metakaggle_interview.Competitions`

WHERE HostSegmentTitle IN ('InClass', 'Featured', 'Research')

ORDER BY HostSegmentTitle, Year ASC

"""

query_job = bigquery_client.query(query)  # Make an API request.

df = query_job.result().to_dataframe()



fig, ax = plt.subplots(figsize=(15,7))

df.groupby(['Year','HostSegmentTitle']).count()['amount'].unstack().plot(ax=ax)
def plot_year_graph(year):

    query = """

            SELECT FORMAT_DATE("%m", DATE(EnabledDate)) AS month, HostSegmentTitle, 1 as amount from `kaggle-playground-170215.metakaggle_interview.Competitions` 

            WHERE HostSegmentTitle IN ('InClass', 'Featured', 'Research') 

                 AND FORMAT_DATE("%Y", DATE(EnabledDate)) = '{0}'

            -- GROUP BY month, HostSegmentTitle

            ORDER BY month ASC, HostSegmentTitle ASC

        """.format(year)

    query_job = bigquery_client.query(query)  # Make an API request.

    

    df = query_job.result().to_dataframe()

    fig, ax = plt.subplots(figsize=(15,7))

    plt.title(year)

    

    newdf = df.groupby(['month','HostSegmentTitle']).count()['amount'].unstack()

    newdf = newdf.replace(np.nan, 0)

    newdf.plot(ax=ax)



years = [2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]

for year in years:

    plot_year_graph(year)
query = """  

    SELECT 

        FORMAT_DATE("%Y", DATE(EnabledDate)) AS Year,

        SUM(TotalTeams) AS TotalTeam,

        SUM(IF(HasKernels IS TRUE, TotalTeams, 0)) AS TotalTeam_Kernel,

        SUM(IF(HasKernels IS NOT TRUE, TotalTeams, 0)) AS TotalTeam_NoKernel

    FROM `kaggle-playground-170215.metakaggle_interview.Competitions`

    GROUP BY Year

    ORDER BY Year DESC 

"""

query_job = bigquery_client.query(query)  # Make an API request.

query_job.result().to_dataframe()
query = """  

    SELECT 

        FORMAT_DATE("%Y", DATE(EnabledDate)) AS Year,

        SUM(TotalSubmissions) AS TotalSubmissions,

        SUM(IF(OnlyAllowKernelSubmissions IS TRUE, TotalSubmissions, 0)) AS TotalSubmissions_OnlyAllowKernels,

        SUM(IF(OnlyAllowKernelSubmissions IS NOT TRUE, TotalSubmissions, 0)) AS TotalSubmissions_NotOnlyAllowKernels

    FROM `kaggle-playground-170215.metakaggle_interview.Competitions`

    GROUP BY Year

    ORDER BY Year DESC 

"""

query_job = bigquery_client.query(query)  # Make an API request.

query_job.result().to_dataframe()