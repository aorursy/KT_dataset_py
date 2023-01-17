def ETLfromURL(url, filepath, sep=',', delim='\n'): 
    #Use r"filepath name" if you have issues
    
    #Import libraries to use 
    import pandas as pd
    import urllib3
    import os
    
    #Stablish a web connection with ul3 library
    http = urllib3.PoolManager()
    r = http.request('GET',url)
    object_data = r.data
    
    #Convert a binary object (Bytes) to String using UTF-8 encoding
    data = object_data.decode('utf-8')
    
    #Split the string into array of rows with delim parameter
    lines = data.split(delim)
    
    #Extract first row as header with sep parameter
    headers = lines[0].split(sep)
    
    #Create an empty dictionary where the processed data will go
    counter = 0; dict = {}
    for column in headers:
        dict[column]=[]
    
    #Filling dictionary with row by row data processing 
    for row in lines:
        if(counter > 0):
            values = row.strip().split(sep)
            for i in range(len(headers)):
                dict[headers[i]].append(values[i])
        counter += 1
    
    #Print row count 
    print("Dataset has %d rows"%(counter))
    
    #Convert dictionary into a dataframe and verify its data
    df = pd.DataFrame(dict)
    print(df.head())
    
    #Save file with different extensions in a given directory (if you want a SQL database, you need import mysql.connector)
    path = os.path.join(filepath,'Data_from_URL')
    df.to_csv(path+".csv")
    df.to_excel(path+".xlsx")
    df.to_html(path+".html")

    return