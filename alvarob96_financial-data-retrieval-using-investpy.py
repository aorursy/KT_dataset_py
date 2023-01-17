!pip install git+https://github.com/alvarobartt/investpy.git@developer
import investpy
recent_data = investpy.get_stock_recent_data(stock='AAPL', country='United States')



recent_data.head()
historical_data = investpy.get_stock_historical_data(stock='AAPL', country='United States', from_date='01/01/2019', to_date='01/01/2020')



historical_data.head()
stock_information = investpy.get_stock_information(stock='AAPL', country='United States', as_json=True)



stock_information
search_results = investpy.search_quotes(text='Apple', products=['stocks'], countries=['United States'])
for search_result in search_results:

    print(search_result)
search_result = search_results.pop(0)

print(search_result)
recent_data = search_result.retrieve_recent_data()



recent_data.head()
historical_data = search_result.retrieve_historical_data(from_date='01/01/2019', to_date='01/01/2020')



historical_data.head()
stock_information = search_result.retrieve_information()



stock_information