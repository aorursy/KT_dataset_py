import sqlite3
import pandas as pd


conn = sqlite3.connect('../input/ffffff/ff.db')
c = conn.cursor()
for row in c.execute(
                    # SQL statement 
                    """
                        SELECT   * 
                        FROM     Cars 
                        
                     """ ):
    print(row)
for row in c.execute(
                    # SQL statement 
                    """
                        SELECT   ParkingNumber 
                        FROM     ParkingPlaces 
                        
                     """ ):
    print(row)
for row in c.execute(
                    # SQL statement 
                    """
                        SELECT   * 
                        FROM     CarsParkings  
                        
                     """ ):
    print(row)
for row in c.execute(
                    # SQL statement 
                    """
                        SELECT Cars.CarNumber , ParkingPlaces.ParkingNumber
                        FROM Cars
                        JOIN CarsParkings 
                        ON CarsParkings.CarID = Cars.id
                        JOIN ParkingPlaces 
                        ON ParkingPlaces.id = CarsParkings.ParkingID
                     """ ):
    print(row)