import bq_helper
Traffic_Fatalities = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")
AccidentCountByHourQuery =     """ 
                                SELECT  DATE(timestamp_of_crash),
                                        EXTRACT(HOUR FROM timestamp_of_crash) AS `Hour`,
                                        COUNT(consecutive_number) AS `Accidents`
                                FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
                                GROUP BY  timestamp_of_crash
                                ORDER BY  DATE(timestamp_of_crash) DESC,COUNT(consecutive_number) DESC                                        
                               """
Traffic_Fatalities.query_to_pandas_safe(AccidentCountByHourQuery)          
HitAndRunsByStateQuery =     """ 
                                SELECT  registration_state_name AS `State`,
                                                COUNT(consecutive_number) AS `HitAndRuns`
                                FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
                                WHERE hit_and_run LIKE "Yes"
                                GROUP BY  registration_state_name
                                ORDER BY  COUNT(consecutive_number) DESC                                        
                               """
Traffic_Fatalities.query_to_pandas_safe(HitAndRunsByStateQuery)          
HitAndRunsByStateQuery1 =     """ 
                                SELECT registration_state_name AS `State`,
                                                COUNT(consecutive_number) AS `HitAndRuns`
                                FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
                                WHERE hit_and_run LIKE "Yes"
                                GROUP BY  registration_state_name
                                ORDER BY  COUNT(consecutive_number) DESC                                        
                                LIMIT 1
                               """
Traffic_Fatalities.query_to_pandas_safe(HitAndRunsByStateQuery1)   
HitAndRunsByStateQuery1 =     """ 
                                SELECT registration_state_name AS `State`,
                                                COUNT(consecutive_number) AS `HitAndRuns`
                                FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
                                WHERE hit_and_run LIKE "Yes" AND registration_state_name NOT LIKE "Unknown"
                                GROUP BY  registration_state_name
                                ORDER BY  COUNT(consecutive_number) DESC                                        
                                LIMIT 1
                               """
Traffic_Fatalities.query_to_pandas_safe(HitAndRunsByStateQuery1)   