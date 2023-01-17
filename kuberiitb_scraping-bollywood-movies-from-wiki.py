from requests import get

from bs4 import BeautifulSoup

from datetime import datetime
def getMovieForYear(year = 2019):

    """

    Scrape movies for a given year

    Parse the page using beautifulsoup and call getMovieForQuarter() with data for each quarter, 

    Merge the output and returns final whole year data.

    """

    url = 'https://en.wikipedia.org/wiki/List_of_Bollywood_films_of_{}'.format(year)

    print("parsing {}".format(url))

    response = get(url)

    html_soup = BeautifulSoup(response.text, 'html.parser')

    movie_containers = html_soup.find_all('tbody')

    moviesList=[]

    for movie_container in movie_containers:

        try:

            moviesList.append(getMovieForQuarter(movie_container))

            

        except:

            pass

    if len(moviesList)>0: 

        moviesDF = pd.concat(moviesList)

        return moviesDF

    else:

        return None
def getMovieForQuarter(container):

    """

    Scrape movies information for given data for a quarter and return a pandas dataframe

    """

    dataList = []

    for movies in container:

        if movies.find('td'):

            #print("START")

            #print(movies)

            #print("END")



            try:

                data = {}

                counter=0

                #style="text-align:center;background:#f1daf1;" day

                #style="text-align:center; background:plum; textcolor:#000; month

                for cols in movies.find_all('td'):

                    counter+=1

                    if counter not in data:data[counter]=[]

                    if cols.find('ul') is not None:

                        for values in cols.ul:

                            #print(counter, values.text)

                            data[counter].append(values.text.strip())

                    else:

                        #print(counter, cols.text)

                        data[counter].append(cols.text.strip())

                start=1

                if len(data)==7:

                    month=data[start]

                    day=data[start+1]

                    start+=2

                elif len(data)==6:

                    day=data[start]

                    start+=1

                #print(start)

                dataDict  = {'month':month[0],

                               'date':day[0],

                                'title':data[start][0],

                                'director':data[start+1][0],

                                'cast':';'.join(data[start+2]),

                                'production':data[start+3][0],

                                 'len': len(data)

                                }

                #print(data)

                dataList.append(dataDict)

            except Exception as e: 

                print("Err",e)

    #del month, day

    

    moviesDF = pd.DataFrame(dataList)

    moviesDF['release_date']= moviesDF.apply(lambda x:datetime.strptime('{} {} {}'.format(x['date'].zfill(2), x['month'],year), '%d %b %Y'),axis=1)

    moviesDF.drop(columns=['month','date'],inplace=True)

    #moviesDF.head()

    return moviesDF
for year in range(2019,2012,-1):

    print(year)

    movies_out = getMovieForYear(year=year)

    print(movies_out.head())

    if movies_out is not None: 

        print(movies_out.shape)

        movies_out['release_date'].hist()
getMovieForYear(year=2020)
debugging = """

    year = 2012

    url = 'https://en.wikipedia.org/wiki/List_of_Bollywood_films_of_{}'.format(year)

    print("parsing {}".format(url))

    response = get(url)

    html_soup = BeautifulSoup(response.text, 'html.parser')

    movie_containers = html_soup.find_all('tbody')

    print(movie_containers[3])

    """
#debugging

x = """

    container = movie_containers[3]

    dataList = []

    for movies in container:

        if movies.find('td'):

            #print("START")

            #print(movies)

            #print("END")



            try:

                data = {}

                counter=0

                #style="text-align:center;background:#f1daf1;" day

                #style="text-align:center; background:plum; textcolor:#000; month

                for cols in movies.find_all('td'):

                    counter+=1

                    if counter not in data:data[counter]=[]

                    if cols.find('ul') is not None:

                        for values in cols.ul:

                            #print(counter, values.text)

                            data[counter].append(values.text.strip())

                    else:

                        #print(counter, cols.text)

                        data[counter].append(cols.text.strip())

                start=1

                if len(data)==7:

                    month=data[start]

                    day=data[start+1]

                    start+=2

                elif len(data)==6:

                    day=data[start]

                    start+=1

                #print(start)

                dataDict  = {'month':month[0],

                               'date':day[0],

                                'title':data[start][0],

                                'director':data[start+1][0],

                                'cast':';'.join(data[start+2]),

                                'production':data[start+3][0],

                                 'len': len(data)

                                }

                #print(data)

                dataList.append(dataDict)

            except Exception as e: 

                pass #print("Err",e)

    #del month, day

    

    moviesDF = pd.DataFrame(dataList)

    print(moviesDF.head())

    moviesDF['release_date']= moviesDF.apply(lambda x:datetime.strptime('{} {} {}'.format(x['date'].zfill(2), x['month'],year), '%d %b %Y'),axis=1)

    moviesDF.drop(columns=['month','date'],inplace=True)

    moviesDF.head()

    print(moviesDF.shape)

    #return moviesDF

"""