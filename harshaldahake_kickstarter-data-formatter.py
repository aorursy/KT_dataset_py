import csv
with open('2018_kickstarter_modified.csv', 'r') as oldData:
    with open('2018_kickstarter_modified_new.csv', 'w', newline='') as newData:
        csvReader = csv.reader(oldData)
        categorySet = set()
        countrySet = set()
        resultList = ['failed', 'successful']
        
        next(csvReader)
        for row in csvReader:
            categorySet.add(row[0])
            countrySet.add(row[8])
        """
        categoryDict = {}
        for i, category in enumerate(categorySet):
            categoryDict[category] = [0 for x in range(i)] + [1] + [0 for x in range(i + 1, len(categorySet))]
                                                                                                                                   
        countryDict = {}
        for i, country in enumerate(countrySet):
            countryDict[country] = [0 for x in range(i)] + [1] + [0 for x in range(i + 1, len(countrySet))]
        """    
        oldData.seek(0)
        csvWriter = csv.writer(newData)
        
        csvWriter.writerow(next(csvReader, 0) + ["categories_encoded"] + ["countries_encoded"] + ["result_encoded"])
        
        for row in csvReader:
            csvWriter.writerow(row + [list(categorySet).index(row[0])] + [list(countrySet).index(row[8])] + ([resultList.index(row[4])] if row[4] in resultList else [-1]))
                                                                                                                                                                                
        print(categorySet)
        print(countrySet)
            
        
            

