# https://www.kaggle.com/LondonDataStore/london-crime

import pandas as pd

# https://github.com/SohierDane/BigQuery_Helper

from bq_helper import BigQueryHelper



bq_assistant = BigQueryHelper("bigquery-public-data", "london_crime")
query = """

       SELECT b1 AS Borough,

       Violence_Against_the_Person,

       Theft_and_Handling,

       Drugs,

       Other_Notifiable_Offences,

       Robbery,

       Criminal_Damage,

       Burglary

FROM (((((((

              (SELECT borough AS b1,

                      sum(value) Theft_and_Handling

               FROM `bigquery-public-data.london_crime.crime_by_lsoa`

               WHERE major_category = "Violence Against the Person"

                 AND YEAR = 2008

               GROUP BY borough)

            JOIN

              (SELECT borough AS b2,

                      sum(value) Violence_Against_the_Person

               FROM `bigquery-public-data.london_crime.crime_by_lsoa`

               WHERE major_category = "Theft and Handling"

                 AND YEAR = 2008

               GROUP BY borough) ON b1 = b2)

           JOIN

             (SELECT borough AS b3,

                     sum(value) Drugs

              FROM `bigquery-public-data.london_crime.crime_by_lsoa`

              WHERE major_category = "Drugs"

                AND YEAR = 2008

              GROUP BY borough) ON b2 = b3)

          JOIN

            (SELECT borough AS b4,

                    sum(value) Other_Notifiable_Offences

             FROM `bigquery-public-data.london_crime.crime_by_lsoa`

             WHERE major_category = "Other Notifiable Offences"

               AND YEAR = 2008

             GROUP BY borough) ON b3 = b4)

         JOIN

           (SELECT borough AS b5,

                   sum(value) Robbery

            FROM `bigquery-public-data.london_crime.crime_by_lsoa`

            WHERE major_category = "Robbery"

              AND YEAR = 2008

            GROUP BY borough) ON b4 = b5)

        JOIN

          (SELECT borough AS b6,

                  sum(value) Criminal_Damage

           FROM `bigquery-public-data.london_crime.crime_by_lsoa`

           WHERE major_category = "Criminal Damage"

             AND YEAR = 2008

           GROUP BY borough) ON b5 = b6)

       JOIN

         (SELECT borough AS b7,

                 sum(value) Burglary

          FROM `bigquery-public-data.london_crime.crime_by_lsoa`

          WHERE major_category = "Burglary"

            AND YEAR = 2008

          GROUP BY borough) ON b6 = b7)

      JOIN

        (SELECT borough AS b8,

                sum(value) Sexual_Offences

         FROM `bigquery-public-data.london_crime.crime_by_lsoa`

         WHERE major_category = "Sexual Offences"

           AND YEAR = 2008

         GROUP BY borough) ON b7 = b8)

JOIN

  (SELECT borough AS b9,

          sum(value) Fraud_or_Forgery

   FROM `bigquery-public-data.london_crime.crime_by_lsoa`

   WHERE major_category = "Fraud or Forgery"

     AND YEAR = 2008

   GROUP BY borough) ON b8 = b9

ORDER BY Borough

"""



two_thousand_and_eight = bq_assistant.query_to_pandas_safe(query)



# two_thousand_and_eight_inverted = two_thousand_and_eight.transpose()



crimes_matrix = two_thousand_and_eight.to_numpy()



# Každý riadok predeliť populáciou v danom roku

population_2008 = [172452, 339212, 226652, 290901, 304968, 210273, 349308, 324022, 297443, 239748, 231041, 177088, 244459, 229567, 231793, 261051, 237907, 192089, 162579, 156027, 289126, 266508, 195859, 276478, 265452, 182927, 276973, 185860, 231893, 242098, 294305, 218673]



for i in range(len(crimes_matrix)):

    for j in range(len(crimes_matrix[0])):

        if (j == 0):

            continue

        crimes_matrix[i][j] /= population_2008[i]

    

# Každý stĺpec predeliť maximom z daného stĺpca

table_divided_by_population = pd.DataFrame(crimes_matrix) 

max_values = table_divided_by_population.max()



for j in range(len(crimes_matrix[0])):

    if (j == 0):

        continue

    for i in range(len(crimes_matrix)):

        crimes_matrix[i][j] /= max_values[j]



table_divided_by_max_values = pd.DataFrame(crimes_matrix) 

table_divided_by_max_values
query = """

       SELECT b1 AS Borough,

       Violence_Against_the_Person,

       Theft_and_Handling,

       Drugs,

       Other_Notifiable_Offences,

       Robbery,

       Criminal_Damage,

       Burglary

FROM (((((((

              (SELECT borough AS b1,

                      sum(value) Theft_and_Handling

               FROM `bigquery-public-data.london_crime.crime_by_lsoa`

               WHERE major_category = "Violence Against the Person"

                 AND YEAR = 2009

               GROUP BY borough)

            JOIN

              (SELECT borough AS b2,

                      sum(value) Violence_Against_the_Person

               FROM `bigquery-public-data.london_crime.crime_by_lsoa`

               WHERE major_category = "Theft and Handling"

                 AND YEAR = 2009

               GROUP BY borough) ON b1 = b2)

           JOIN

             (SELECT borough AS b3,

                     sum(value) Drugs

              FROM `bigquery-public-data.london_crime.crime_by_lsoa`

              WHERE major_category = "Drugs"

                AND YEAR = 2009

              GROUP BY borough) ON b2 = b3)

          JOIN

            (SELECT borough AS b4,

                    sum(value) Other_Notifiable_Offences

             FROM `bigquery-public-data.london_crime.crime_by_lsoa`

             WHERE major_category = "Other Notifiable Offences"

               AND YEAR = 2009

             GROUP BY borough) ON b3 = b4)

         JOIN

           (SELECT borough AS b5,

                   sum(value) Robbery

            FROM `bigquery-public-data.london_crime.crime_by_lsoa`

            WHERE major_category = "Robbery"

              AND YEAR = 2009

            GROUP BY borough) ON b4 = b5)

        JOIN

          (SELECT borough AS b6,

                  sum(value) Criminal_Damage

           FROM `bigquery-public-data.london_crime.crime_by_lsoa`

           WHERE major_category = "Criminal Damage"

             AND YEAR = 2009

           GROUP BY borough) ON b5 = b6)

       JOIN

         (SELECT borough AS b7,

                 sum(value) Burglary

          FROM `bigquery-public-data.london_crime.crime_by_lsoa`

          WHERE major_category = "Burglary"

            AND YEAR = 2009

          GROUP BY borough) ON b6 = b7)

      JOIN

        (SELECT borough AS b8,

                sum(value) Sexual_Offences

         FROM `bigquery-public-data.london_crime.crime_by_lsoa`

         WHERE major_category = "Sexual Offences"

           AND YEAR = 2009

         GROUP BY borough) ON b7 = b8)

JOIN

  (SELECT borough AS b9,

          sum(value) Fraud_or_Forgery

   FROM `bigquery-public-data.london_crime.crime_by_lsoa`

   WHERE major_category = "Fraud or Forgery"

     AND YEAR = 2009

   GROUP BY borough) ON b8 = b9

ORDER BY Borough

"""



twoThousandAndNine = bq_assistant.query_to_pandas_safe(query)



crimes_matrix = twoThousandAndNine.to_numpy()



# Každý riadok predeliť populáciou v danom roku

population_2009 = [177580,345829,228146,298118,306924,212924,335094,335094,301971,243672,236622,180116,249805,233495,234127,265665,243366,196704,161893,157307,294050,270418,198136,286447,270435,184394,281120,188167,240495,248140,299347,216980]



for i in range(len(crimes_matrix)):

    for j in range(len(crimes_matrix[0])):

        if (j == 0):

            continue

        crimes_matrix[i][j] /= population_2009[i]

    

# Každý stĺpec predeliť maximom z daného stĺpca

table_divided_by_population = pd.DataFrame(crimes_matrix) 

max_values = table_divided_by_population.max()



for j in range(len(crimes_matrix[0])):

    if (j == 0):

        continue

    for i in range(len(crimes_matrix)):

        crimes_matrix[i][j] /= max_values[j]



table_divided_by_max_values = pd.DataFrame(crimes_matrix) 

table_divided_by_max_values
query = """

       SELECT b1 AS Borough,

       Violence_Against_the_Person,

       Theft_and_Handling,

       Drugs,

       Other_Notifiable_Offences,

       Robbery,

       Criminal_Damage,

       Burglary

FROM (((((((

              (SELECT borough AS b1,

                      sum(value) Theft_and_Handling

               FROM `bigquery-public-data.london_crime.crime_by_lsoa`

               WHERE major_category = "Violence Against the Person"

                 AND YEAR = 2010

               GROUP BY borough)

            JOIN

              (SELECT borough AS b2,

                      sum(value) Violence_Against_the_Person

               FROM `bigquery-public-data.london_crime.crime_by_lsoa`

               WHERE major_category = "Theft and Handling"

                 AND YEAR = 2010

               GROUP BY borough) ON b1 = b2)

           JOIN

             (SELECT borough AS b3,

                     sum(value) Drugs

              FROM `bigquery-public-data.london_crime.crime_by_lsoa`

              WHERE major_category = "Drugs"

                AND YEAR = 2010

              GROUP BY borough) ON b2 = b3)

          JOIN

            (SELECT borough AS b4,

                    sum(value) Other_Notifiable_Offences

             FROM `bigquery-public-data.london_crime.crime_by_lsoa`

             WHERE major_category = "Other Notifiable Offences"

               AND YEAR = 2010

             GROUP BY borough) ON b3 = b4)

         JOIN

           (SELECT borough AS b5,

                   sum(value) Robbery

            FROM `bigquery-public-data.london_crime.crime_by_lsoa`

            WHERE major_category = "Robbery"

              AND YEAR = 2010

            GROUP BY borough) ON b4 = b5)

        JOIN

          (SELECT borough AS b6,

                  sum(value) Criminal_Damage

           FROM `bigquery-public-data.london_crime.crime_by_lsoa`

           WHERE major_category = "Criminal Damage"

             AND YEAR = 2010

           GROUP BY borough) ON b5 = b6)

       JOIN

         (SELECT borough AS b7,

                 sum(value) Burglary

          FROM `bigquery-public-data.london_crime.crime_by_lsoa`

          WHERE major_category = "Burglary"

            AND YEAR = 2010

          GROUP BY borough) ON b6 = b7)

      JOIN

        (SELECT borough AS b8,

                sum(value) Sexual_Offences

         FROM `bigquery-public-data.london_crime.crime_by_lsoa`

         WHERE major_category = "Sexual Offences"

           AND YEAR = 2010

         GROUP BY borough) ON b7 = b8)

JOIN

  (SELECT borough AS b9,

          sum(value) Fraud_or_Forgery

   FROM `bigquery-public-data.london_crime.crime_by_lsoa`

   WHERE major_category = "Fraud or Forgery"

     AND YEAR = 2010

   GROUP BY borough) ON b8 = b9

ORDER BY Borough

"""



twoThousandAndTen = bq_assistant.query_to_pandas_safe(query)



crimes_matrix = twoThousandAndTen.to_numpy()



# Každý riadok predeliť populáciou v danom roku

population_2010 = [182838,351438,13062,304785,308560,77010,357951,334073,307648,249171,241739,180842,252742,237451,236234,269465,249236,200129,160463,158648,297650,272525,199136,299171,275088,186304,283777,189321,248520,254009,302620,217187]



for i in range(len(crimes_matrix)):

    for j in range(len(crimes_matrix[0])):

        if (j == 0):

            continue

        crimes_matrix[i][j] /= population_2010[i]

    

# Každý stĺpec predeliť maximom z daného stĺpca

table_divided_by_population = pd.DataFrame(crimes_matrix) 

max_values = table_divided_by_population.max()



for j in range(len(crimes_matrix[0])):

    if (j == 0):

        continue

    for i in range(len(crimes_matrix)):

        crimes_matrix[i][j] /= max_values[j]



table_divided_by_max_values = pd.DataFrame(crimes_matrix) 

table_divided_by_max_values
query = """

       SELECT b1 AS Borough,

       Violence_Against_the_Person,

       Theft_and_Handling,

       Drugs,

       Other_Notifiable_Offences,

       Robbery,

       Criminal_Damage,

       Burglary

FROM (((((((

              (SELECT borough AS b1,

                      sum(value) Theft_and_Handling

               FROM `bigquery-public-data.london_crime.crime_by_lsoa`

               WHERE major_category = "Violence Against the Person"

                 AND YEAR = 2011

               GROUP BY borough)

            JOIN

              (SELECT borough AS b2,

                      sum(value) Violence_Against_the_Person

               FROM `bigquery-public-data.london_crime.crime_by_lsoa`

               WHERE major_category = "Theft and Handling"

                 AND YEAR = 2011

               GROUP BY borough) ON b1 = b2)

           JOIN

             (SELECT borough AS b3,

                     sum(value) Drugs

              FROM `bigquery-public-data.london_crime.crime_by_lsoa`

              WHERE major_category = "Drugs"

                AND YEAR = 2011

              GROUP BY borough) ON b2 = b3)

          JOIN

            (SELECT borough AS b4,

                    sum(value) Other_Notifiable_Offences

             FROM `bigquery-public-data.london_crime.crime_by_lsoa`

             WHERE major_category = "Other Notifiable Offences"

               AND YEAR = 2011

             GROUP BY borough) ON b3 = b4)

         JOIN

           (SELECT borough AS b5,

                   sum(value) Robbery

            FROM `bigquery-public-data.london_crime.crime_by_lsoa`

            WHERE major_category = "Robbery"

              AND YEAR = 2011

            GROUP BY borough) ON b4 = b5)

        JOIN

          (SELECT borough AS b6,

                  sum(value) Criminal_Damage

           FROM `bigquery-public-data.london_crime.crime_by_lsoa`

           WHERE major_category = "Criminal Damage"

             AND YEAR = 2011

           GROUP BY borough) ON b5 = b6)

       JOIN

         (SELECT borough AS b7,

                 sum(value) Burglary

          FROM `bigquery-public-data.london_crime.crime_by_lsoa`

          WHERE major_category = "Burglary"

            AND YEAR = 2011

          GROUP BY borough) ON b6 = b7)

      JOIN

        (SELECT borough AS b8,

                sum(value) Sexual_Offences

         FROM `bigquery-public-data.london_crime.crime_by_lsoa`

         WHERE major_category = "Sexual Offences"

           AND YEAR = 2011

         GROUP BY borough) ON b7 = b8)

JOIN

  (SELECT borough AS b9,

          sum(value) Fraud_or_Forgery

   FROM `bigquery-public-data.london_crime.crime_by_lsoa`

   WHERE major_category = "Fraud or Forgery"

     AND YEAR = 2011

   GROUP BY borough) ON b8 = b9

ORDER BY Borough

"""



twentyEleven = bq_assistant.query_to_pandas_safe(query)



crimes_matrix = twentyEleven.to_numpy()



# Každý riadok predeliť populáciou v danom roku

population_2011 = [185911,356386,231997,311215,309392,220338,363378,338449,312466,254557,246270,182493,254926,239056,237232,273936,253957,206125,158649,160060,303086,275885,199693,307984,278970,186990,288283,190146,254096,258249,306995,219396]



for i in range(len(crimes_matrix)):

    for j in range(len(crimes_matrix[0])):

        if (j == 0):

            continue

        crimes_matrix[i][j] /= population_2011[i]

    

# Každý stĺpec predeliť maximom z daného stĺpca

table_divided_by_population = pd.DataFrame(crimes_matrix) 

max_values = table_divided_by_population.max()



for j in range(len(crimes_matrix[0])):

    if (j == 0):

        continue

    for i in range(len(crimes_matrix)):

        crimes_matrix[i][j] /= max_values[j]



table_divided_by_max_values = pd.DataFrame(crimes_matrix) 

table_divided_by_max_values
query = """

       SELECT b1 AS Borough,

       Violence_Against_the_Person,

       Theft_and_Handling,

       Drugs,

       Other_Notifiable_Offences,

       Robbery,

       Criminal_Damage,

       Burglary

FROM (((((((

              (SELECT borough AS b1,

                      sum(value) Theft_and_Handling

               FROM `bigquery-public-data.london_crime.crime_by_lsoa`

               WHERE major_category = "Violence Against the Person"

                 AND YEAR = 2012

               GROUP BY borough)

            JOIN

              (SELECT borough AS b2,

                      sum(value) Violence_Against_the_Person

               FROM `bigquery-public-data.london_crime.crime_by_lsoa`

               WHERE major_category = "Theft and Handling"

                 AND YEAR = 2012

               GROUP BY borough) ON b1 = b2)

           JOIN

             (SELECT borough AS b3,

                     sum(value) Drugs

              FROM `bigquery-public-data.london_crime.crime_by_lsoa`

              WHERE major_category = "Drugs"

                AND YEAR = 2012

              GROUP BY borough) ON b2 = b3)

          JOIN

            (SELECT borough AS b4,

                    sum(value) Other_Notifiable_Offences

             FROM `bigquery-public-data.london_crime.crime_by_lsoa`

             WHERE major_category = "Other Notifiable Offences"

               AND YEAR = 2012

             GROUP BY borough) ON b3 = b4)

         JOIN

           (SELECT borough AS b5,

                   sum(value) Robbery

            FROM `bigquery-public-data.london_crime.crime_by_lsoa`

            WHERE major_category = "Robbery"

              AND YEAR = 2012

            GROUP BY borough) ON b4 = b5)

        JOIN

          (SELECT borough AS b6,

                  sum(value) Criminal_Damage

           FROM `bigquery-public-data.london_crime.crime_by_lsoa`

           WHERE major_category = "Criminal Damage"

             AND YEAR = 2012

           GROUP BY borough) ON b5 = b6)

       JOIN

         (SELECT borough AS b7,

                 sum(value) Burglary

          FROM `bigquery-public-data.london_crime.crime_by_lsoa`

          WHERE major_category = "Burglary"

            AND YEAR = 2012

          GROUP BY borough) ON b6 = b7)

      JOIN

        (SELECT borough AS b8,

                sum(value) Sexual_Offences

         FROM `bigquery-public-data.london_crime.crime_by_lsoa`

         WHERE major_category = "Sexual Offences"

           AND YEAR = 2012

         GROUP BY borough) ON b7 = b8)

JOIN

  (SELECT borough AS b9,

          sum(value) Fraud_or_Forgery

   FROM `bigquery-public-data.london_crime.crime_by_lsoa`

   WHERE major_category = "Fraud or Forgery"

     AND YEAR = 2012

   GROUP BY borough) ON b8 = b9

ORDER BY Borough

"""



twentyTwelve = bq_assistant.query_to_pandas_safe(query)



crimes_matrix = twentyTwelve.to_numpy()



# Každý riadok predeliť populáciou v danom roku

population_2012 = [194352,369088,236687,317264,317899,229719,372752,342494,320524,264008,257379,178685,263386,243373,242080,286808,262407,215671,155594,166793,314242,286180,203223,318227,288272,191365,298465,195914,272890,265797,310516,226841]



for i in range(len(crimes_matrix)):

    for j in range(len(crimes_matrix[0])):

        if (j == 0):

            continue

        crimes_matrix[i][j] /= population_2012[i]

    

# Každý stĺpec predeliť maximom z daného stĺpca

table_divided_by_population = pd.DataFrame(crimes_matrix) 

max_values = table_divided_by_population.max()



for j in range(len(crimes_matrix[0])):

    if (j == 0):

        continue

    for i in range(len(crimes_matrix)):

        crimes_matrix[i][j] /= max_values[j]



table_divided_by_max_values = pd.DataFrame(crimes_matrix) 

table_divided_by_max_values
query = """

       SELECT b1 AS Borough,

       Violence_Against_the_Person,

       Theft_and_Handling,

       Drugs,

       Other_Notifiable_Offences,

       Robbery,

       Criminal_Damage,

       Burglary

FROM (((((((

              (SELECT borough AS b1,

                      sum(value) Theft_and_Handling

               FROM `bigquery-public-data.london_crime.crime_by_lsoa`

               WHERE major_category = "Violence Against the Person"

                 AND YEAR = 2013

               GROUP BY borough)

            JOIN

              (SELECT borough AS b2,

                      sum(value) Violence_Against_the_Person

               FROM `bigquery-public-data.london_crime.crime_by_lsoa`

               WHERE major_category = "Theft and Handling"

                 AND YEAR = 2013

               GROUP BY borough) ON b1 = b2)

           JOIN

             (SELECT borough AS b3,

                     sum(value) Drugs

              FROM `bigquery-public-data.london_crime.crime_by_lsoa`

              WHERE major_category = "Drugs"

                AND YEAR = 2013

              GROUP BY borough) ON b2 = b3)

          JOIN

            (SELECT borough AS b4,

                    sum(value) Other_Notifiable_Offences

             FROM `bigquery-public-data.london_crime.crime_by_lsoa`

             WHERE major_category = "Other Notifiable Offences"

               AND YEAR = 2013

             GROUP BY borough) ON b3 = b4)

         JOIN

           (SELECT borough AS b5,

                   sum(value) Robbery

            FROM `bigquery-public-data.london_crime.crime_by_lsoa`

            WHERE major_category = "Robbery"

              AND YEAR = 2013

            GROUP BY borough) ON b4 = b5)

        JOIN

          (SELECT borough AS b6,

                  sum(value) Criminal_Damage

           FROM `bigquery-public-data.london_crime.crime_by_lsoa`

           WHERE major_category = "Criminal Damage"

             AND YEAR = 2013

           GROUP BY borough) ON b5 = b6)

       JOIN

         (SELECT borough AS b7,

                 sum(value) Burglary

          FROM `bigquery-public-data.london_crime.crime_by_lsoa`

          WHERE major_category = "Burglary"

            AND YEAR = 2013

          GROUP BY borough) ON b6 = b7)

      JOIN

        (SELECT borough AS b8,

                sum(value) Sexual_Offences

         FROM `bigquery-public-data.london_crime.crime_by_lsoa`

         WHERE major_category = "Sexual Offences"

           AND YEAR = 2013

         GROUP BY borough) ON b7 = b8)

JOIN

  (SELECT borough AS b9,

          sum(value) Fraud_or_Forgery

   FROM `bigquery-public-data.london_crime.crime_by_lsoa`

   WHERE major_category = "Fraud or Forgery"

     AND YEAR = 2013

   GROUP BY borough) ON b8 = b9

ORDER BY Borough

"""



twentyThirteen = bq_assistant.query_to_pandas_safe(query)



crimes_matrix = twentyThirteen.to_numpy()



# Každý riadok predeliť populáciou v danom roku

population_2013 = [194352,369088,236687,317264,317899,229719,372752,342494,320524,264008,257379,178685,263386,243373,242080,286808,262407,215671,155594,166793,314242,286180,203223,318227,288272,191365,298465,195914,272890,265797,310516,226841]



for i in range(len(crimes_matrix)):

    for j in range(len(crimes_matrix[0])):

        if (j == 0):

            continue

        crimes_matrix[i][j] /= population_2013[i]

    

# Každý stĺpec predeliť maximom z daného stĺpca

table_divided_by_population = pd.DataFrame(crimes_matrix) 

max_values = table_divided_by_population.max()



for j in range(len(crimes_matrix[0])):

    if (j == 0):

        continue

    for i in range(len(crimes_matrix)):

        crimes_matrix[i][j] /= max_values[j]



table_divided_by_max_values = pd.DataFrame(crimes_matrix) 

table_divided_by_max_values
query = """

       SELECT b1 AS Borough,

       Violence_Against_the_Person,

       Theft_and_Handling,

       Drugs,

       Other_Notifiable_Offences,

       Robbery,

       Criminal_Damage,

       Burglary

FROM (((((((

              (SELECT borough AS b1,

                      sum(value) Theft_and_Handling

               FROM `bigquery-public-data.london_crime.crime_by_lsoa`

               WHERE major_category = "Violence Against the Person"

                 AND YEAR = 2014

               GROUP BY borough)

            JOIN

              (SELECT borough AS b2,

                      sum(value) Violence_Against_the_Person

               FROM `bigquery-public-data.london_crime.crime_by_lsoa`

               WHERE major_category = "Theft and Handling"

                 AND YEAR = 2014

               GROUP BY borough) ON b1 = b2)

           JOIN

             (SELECT borough AS b3,

                     sum(value) Drugs

              FROM `bigquery-public-data.london_crime.crime_by_lsoa`

              WHERE major_category = "Drugs"

                AND YEAR = 2014

              GROUP BY borough) ON b2 = b3)

          JOIN

            (SELECT borough AS b4,

                    sum(value) Other_Notifiable_Offences

             FROM `bigquery-public-data.london_crime.crime_by_lsoa`

             WHERE major_category = "Other Notifiable Offences"

               AND YEAR = 2014

             GROUP BY borough) ON b3 = b4)

         JOIN

           (SELECT borough AS b5,

                   sum(value) Robbery

            FROM `bigquery-public-data.london_crime.crime_by_lsoa`

            WHERE major_category = "Robbery"

              AND YEAR = 2014

            GROUP BY borough) ON b4 = b5)

        JOIN

          (SELECT borough AS b6,

                  sum(value) Criminal_Damage

           FROM `bigquery-public-data.london_crime.crime_by_lsoa`

           WHERE major_category = "Criminal Damage"

             AND YEAR = 2014

           GROUP BY borough) ON b5 = b6)

       JOIN

         (SELECT borough AS b7,

                 sum(value) Burglary

          FROM `bigquery-public-data.london_crime.crime_by_lsoa`

          WHERE major_category = "Burglary"

            AND YEAR = 2014

          GROUP BY borough) ON b6 = b7)

      JOIN

        (SELECT borough AS b8,

                sum(value) Sexual_Offences

         FROM `bigquery-public-data.london_crime.crime_by_lsoa`

         WHERE major_category = "Sexual Offences"

           AND YEAR = 2014

         GROUP BY borough) ON b7 = b8)

JOIN

  (SELECT borough AS b9,

          sum(value) Fraud_or_Forgery

   FROM `bigquery-public-data.london_crime.crime_by_lsoa`

   WHERE major_category = "Fraud or Forgery"

     AND YEAR = 2014

   GROUP BY borough) ON b8 = b9

ORDER BY Borough

"""



twentyFourteen = bq_assistant.query_to_pandas_safe(query)



crimes_matrix = twentyFourteen.to_numpy()



# Každý riadok predeliť populáciou v danom roku

population_2014 = [198294,374915,239865,320762,321278,234846,376040,342118,324574,268678,263150,178365,267541,246011,245974,292690,265568,221030,156190,169958,318216,291933,203515,324322,293055,193585,302538,198134,284015,268020,312145,233292]



for i in range(len(crimes_matrix)):

    for j in range(len(crimes_matrix[0])):

        if (j == 0):

            continue

        crimes_matrix[i][j] /= population_2014[i]

    

# Každý stĺpec predeliť maximom z daného stĺpca

table_divided_by_population = pd.DataFrame(crimes_matrix) 

max_values = table_divided_by_population.max()



for j in range(len(crimes_matrix[0])):

    if (j == 0):

        continue

    for i in range(len(crimes_matrix)):

        crimes_matrix[i][j] /= max_values[j]



table_divided_by_max_values = pd.DataFrame(crimes_matrix) 

table_divided_by_max_values
query = """

       SELECT b1 AS Borough,

       Violence_Against_the_Person,

       Theft_and_Handling,

       Drugs,

       Other_Notifiable_Offences,

       Robbery,

       Criminal_Damage,

       Burglary

FROM (((((((

              (SELECT borough AS b1,

                      sum(value) Theft_and_Handling

               FROM `bigquery-public-data.london_crime.crime_by_lsoa`

               WHERE major_category = "Violence Against the Person"

                 AND YEAR = 2015

               GROUP BY borough)

            JOIN

              (SELECT borough AS b2,

                      sum(value) Violence_Against_the_Person

               FROM `bigquery-public-data.london_crime.crime_by_lsoa`

               WHERE major_category = "Theft and Handling"

                 AND YEAR = 2015

               GROUP BY borough) ON b1 = b2)

           JOIN

             (SELECT borough AS b3,

                     sum(value) Drugs

              FROM `bigquery-public-data.london_crime.crime_by_lsoa`

              WHERE major_category = "Drugs"

                AND YEAR = 2015

              GROUP BY borough) ON b2 = b3)

          JOIN

            (SELECT borough AS b4,

                    sum(value) Other_Notifiable_Offences

             FROM `bigquery-public-data.london_crime.crime_by_lsoa`

             WHERE major_category = "Other Notifiable Offences"

               AND YEAR = 2015

             GROUP BY borough) ON b3 = b4)

         JOIN

           (SELECT borough AS b5,

                   sum(value) Robbery

            FROM `bigquery-public-data.london_crime.crime_by_lsoa`

            WHERE major_category = "Robbery"

              AND YEAR = 2015

            GROUP BY borough) ON b4 = b5)

        JOIN

          (SELECT borough AS b6,

                  sum(value) Criminal_Damage

           FROM `bigquery-public-data.london_crime.crime_by_lsoa`

           WHERE major_category = "Criminal Damage"

             AND YEAR = 2015

           GROUP BY borough) ON b5 = b6)

       JOIN

         (SELECT borough AS b7,

                 sum(value) Burglary

          FROM `bigquery-public-data.london_crime.crime_by_lsoa`

          WHERE major_category = "Burglary"

            AND YEAR = 2015

          GROUP BY borough) ON b6 = b7)

      JOIN

        (SELECT borough AS b8,

                sum(value) Sexual_Offences

         FROM `bigquery-public-data.london_crime.crime_by_lsoa`

         WHERE major_category = "Sexual Offences"

           AND YEAR = 2015

         GROUP BY borough) ON b7 = b8)

JOIN

  (SELECT borough AS b9,

          sum(value) Fraud_or_Forgery

   FROM `bigquery-public-data.london_crime.crime_by_lsoa`

   WHERE major_category = "Fraud or Forgery"

     AND YEAR = 2015

   GROUP BY borough) ON b8 = b9

ORDER BY Borough

"""



twentyFifteen = bq_assistant.query_to_pandas_safe(query)



crimes_matrix = twentyFifteen.to_numpy()



# Každý riadok predeliť populáciou v danom roku

population_2015 = [201979,379691,242142,324012,324857,241059,379031,343059,328433,274803,269009,179410,272864,247130,249085,297735,268770,227692,157711,173525,324431,297325,204565,332817,296793,194730,308901,200145,295236,271170,314544,242299] 



for i in range(len(crimes_matrix)):

    for j in range(len(crimes_matrix[0])):

        if (j == 0):

            continue

        crimes_matrix[i][j] /= population_2015[i]

    

# Každý stĺpec predeliť maximom z daného stĺpca

table_divided_by_population = pd.DataFrame(crimes_matrix) 

max_values = table_divided_by_population.max()



for j in range(len(crimes_matrix[0])):

    if (j == 0):

        continue

    for i in range(len(crimes_matrix)):

        crimes_matrix[i][j] /= max_values[j]



table_divided_by_max_values = pd.DataFrame(crimes_matrix) 

table_divided_by_max_values
query = """

       SELECT b1 AS Borough,

       Violence_Against_the_Person,

       Theft_and_Handling,

       Drugs,

       Other_Notifiable_Offences,

       Robbery,

       Criminal_Damage,

       Burglary

FROM (((((((

              (SELECT borough AS b1,

                      sum(value) Theft_and_Handling

               FROM `bigquery-public-data.london_crime.crime_by_lsoa`

               WHERE major_category = "Violence Against the Person"

                 AND YEAR = 2016

               GROUP BY borough)

            JOIN

              (SELECT borough AS b2,

                      sum(value) Violence_Against_the_Person

               FROM `bigquery-public-data.london_crime.crime_by_lsoa`

               WHERE major_category = "Theft and Handling"

                 AND YEAR = 2016

               GROUP BY borough) ON b1 = b2)

           JOIN

             (SELECT borough AS b3,

                     sum(value) Drugs

              FROM `bigquery-public-data.london_crime.crime_by_lsoa`

              WHERE major_category = "Drugs"

                AND YEAR = 2016

              GROUP BY borough) ON b2 = b3)

          JOIN

            (SELECT borough AS b4,

                    sum(value) Other_Notifiable_Offences

             FROM `bigquery-public-data.london_crime.crime_by_lsoa`

             WHERE major_category = "Other Notifiable Offences"

               AND YEAR = 2016

             GROUP BY borough) ON b3 = b4)

         JOIN

           (SELECT borough AS b5,

                   sum(value) Robbery

            FROM `bigquery-public-data.london_crime.crime_by_lsoa`

            WHERE major_category = "Robbery"

              AND YEAR = 2016

            GROUP BY borough) ON b4 = b5)

        JOIN

          (SELECT borough AS b6,

                  sum(value) Criminal_Damage

           FROM `bigquery-public-data.london_crime.crime_by_lsoa`

           WHERE major_category = "Criminal Damage"

             AND YEAR = 2016

           GROUP BY borough) ON b5 = b6)

       JOIN

         (SELECT borough AS b7,

                 sum(value) Burglary

          FROM `bigquery-public-data.london_crime.crime_by_lsoa`

          WHERE major_category = "Burglary"

            AND YEAR = 2016

          GROUP BY borough) ON b6 = b7)

      JOIN

        (SELECT borough AS b8,

                sum(value) Sexual_Offences

         FROM `bigquery-public-data.london_crime.crime_by_lsoa`

         WHERE major_category = "Sexual Offences"

           AND YEAR = 2016

         GROUP BY borough) ON b7 = b8)

JOIN

  (SELECT borough AS b9,

          sum(value) Fraud_or_Forgery

   FROM `bigquery-public-data.london_crime.crime_by_lsoa`

   WHERE major_category = "Fraud or Forgery"

     AND YEAR = 2016

   GROUP BY borough) ON b8 = b9

ORDER BY Borough

"""



twentySixteen = bq_assistant.query_to_pandas_safe(query)



crimes_matrix = twentySixteen.to_numpy()



# Každý riadok predeliť populáciou v danom roku

population_2016 = [206460,386083,244760,328254,326889,246181,382304,343196,331395,279766,273526,179654,278451,248752,252783,302471,271139,232865,156726,176107,327910,301867,205029,340978,299249,195846,313223,202220,304854,275843,316096,247614]



for i in range(len(crimes_matrix)):

    for j in range(len(crimes_matrix[0])):

        if (j == 0):

            continue

        crimes_matrix[i][j] /= population_2016[i]

    

# Každý stĺpec predeliť maximom z daného stĺpca

table_divided_by_population = pd.DataFrame(crimes_matrix) 

max_values = table_divided_by_population.max()



for j in range(len(crimes_matrix[0])):

    if (j == 0):

        continue

    for i in range(len(crimes_matrix)):

        crimes_matrix[i][j] /= max_values[j]



table_divided_by_max_values = pd.DataFrame(crimes_matrix) 

table_divided_by_max_values