from google.cloud import bigquery

import pandas as pd

pd.set_option('display.max_rows', 1000)
bq = bigquery.Client()
sql_query = """

#standardSQL



select

    Mascot_Type,

    avg(Points) Avg_Points

from

(

    select

      coalesce(mascots.mascot_common_name, mascots.non_tax_type, mascots.mascot) as Mascot_Type,

      points_game as Points

    from `bigquery-public-data.ncaa_basketball.mbb_teams_games_sr` as games

    inner join `bigquery-public-data.ncaa_basketball.mascots` as mascots

        on games.team_id = mascots.id

    where

        season > 1939



    union all



    select

      coalesce(mascots.mascot_common_name, mascots.non_tax_type, mascots.mascot) as Mascot_Type,

      opp_points_game as Points

    from `bigquery-public-data.ncaa_basketball.mbb_teams_games_sr` as games

    inner join `bigquery-public-data.ncaa_basketball.mascots` as mascots

        on games.opp_id = mascots.id

    where

        season > 1939

)

group by Mascot_Type

order by Avg_Points desc



"""

results = bq.query(sql_query).to_dataframe()

results
sql_query = """

#standardSQL



select

    mascots.Market,

    mascots.Name,

    mascots.Mascot_Name,

    stats.Mascot_Type,

    stats.Avg_Points

from

    `bigquery-public-data.ncaa_basketball.mascots` as mascots

    inner join

    (

        select

            Mascot_Type,

            avg(Points) Avg_Points

        from

        (

            select

              coalesce(mascots.mascot_common_name, mascots.non_tax_type, mascots.mascot) as Mascot_Type,

              points_game as Points

            from `bigquery-public-data.ncaa_basketball.mbb_teams_games_sr` as games

            inner join `bigquery-public-data.ncaa_basketball.mascots` as mascots

                on games.team_id = mascots.id

            where

                season > 1939



            union all



            select

              coalesce(mascots.mascot_common_name, mascots.non_tax_type, mascots.mascot) as Mascot_Type,

              opp_points_game as Points

            from `bigquery-public-data.ncaa_basketball.mbb_teams_games_sr` as games

            inner join `bigquery-public-data.ncaa_basketball.mascots` as mascots

                on games.opp_id = mascots.id

            where

                season > 1939

        )

        group by Mascot_Type

    ) stats

    on coalesce(mascots.mascot_common_name, mascots.non_tax_type, mascots.mascot) = stats.mascot_type

order by

    stats.Avg_Points desc



"""

results = bq.query(sql_query).to_dataframe()

results
teams = pd.read_csv("../input/teams.csv")

joinedResults = teams.set_index("Team").join(results.set_index("Market")).sort_values("Avg_Points", ascending=False)

joinedResults
newResults = teams.set_index("Team").join(results.set_index("Market"))

newResults = newResults.sort_values("Avg_Points", ascending=False)

newResults.insert(1, 'Mascot Rank', range(1, 1 + len(newResults)))

newResults["Final Score"] = (newResults["Seed"] * 2 + newResults["Mascot Rank"]) / 2

newResults.sort_values("Final Score", ascending=True)