# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from google.cloud import bigquery
compute_alpha = """
#standardSQL
SELECT 
    SAFE_DIVIDE(
    SUM(arrival_delay * departure_delay),
    SUM(departure_delay * departure_delay)) AS alpha
FROM
(
    SELECT
        RAND() AS splitfield,
        arrival_delay,
        departure_delay
    FROM
        `bigquery-samples.airline_ontime_data.flights`
    WHERE
        departure_airport = 'DEN'
        AND arrival_airport = 'LAX'
)
WHERE
    splitfield < 0.8
"""
results = bigquery.Client().query(compute_alpha).to_dataframe()
alpha = results['alpha'][0]
print(alpha)

compute_rmse = """
#standardSQL
SELECT
    dataset,
    SQRT(
        AVG(
            (arrival_delay - ALPHA * departure_delay) *
            (arrival_delay - ALPHA * departure_delay)
        )
    ) AS rmse,
    COUNT(arrival_delay) AS num_flights
FROM (
    SELECT
        IF (RAND() < 0.8, 'train', 'eval') AS dataset,
        arrival_delay,
        departure_delay
    FROM
        `bigquery-samples.airline_ontime_data.flights`
    WHERE
        departure_airport = 'DEN'
        AND arrival_airport = 'LAX' )
GROUP BY
    dataset
"""
bigquery.Client().query(
    compute_rmse.replace('ALPHA', str(alpha))).to_dataframe()
compute_alpha = """
#standardSQL
SELECT 
    SAFE_DIVIDE(
        SUM(arrival_delay * departure_delay),
        SUM(departure_delay * departure_delay)) AS alpha
FROM
    `bigquery-samples.airline_ontime_data.flights`
WHERE
    departure_airport = 'DEN'
    AND arrival_airport = 'LAX'
    AND ABS(MOD(FARM_FINGERPRINT(date), 10)) < 8
    
"""
results = bigquery.Client().query(compute_alpha).to_dataframe()
alpha = results['alpha'][0]
print(alpha)
query_alpha = """
#standardSQL
SELECT date, abs(mod(farm_fingerprint(date),10)) as hash_value,
    SAFE_DIVIDE(
        SUM(arrival_delay * departure_delay),
        SUM(departure_delay * departure_delay)) AS alpha
FROM
    `bigquery-samples.airline_ontime_data.flights`
WHERE
    departure_airport = 'DEN'
    AND arrival_airport = 'LAX'
    AND ABS(MOD(FARM_FINGERPRINT(date), 10)) < 8
    GROUP BY DATE,HASH_VALUE
"""
results = bigquery.Client().query(query_alpha).to_dataframe()
print(results.head(10))
compute_rmse = """
#standardSQL
SELECT
    IF(ABS(MOD(FARM_FINGERPRINT(date), 10)) < 8, 'train', 'eval') AS dataset,
    SQRT(
        AVG(
            (arrival_delay - ALPHA * departure_delay) *
            (arrival_delay - ALPHA * departure_delay)
        )
    ) AS rmse,
    COUNT(arrival_delay) AS num_flights
FROM
    `bigquery-samples.airline_ontime_data.flights`
WHERE
    departure_airport = 'DEN'
    AND arrival_airport = 'LAX'
GROUP BY
    dataset
"""
print(bigquery.Client().query(
    compute_rmse.replace('ALPHA', str(alpha))).to_dataframe().head())
