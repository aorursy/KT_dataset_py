%%time

import time

for i in range(10):

    the_sum = sum([j*j for j in range(1000000)])

    time.sleep(1)
import time

start_time = time.time()

start_cpu_time = time.clock()

for i in range(10):

    the_sum = sum([j*j for j in range(1000000)])

    time.sleep(1)



print(f'CPU time {time.clock() - start_cpu_time}')

print(f'Wall time {time.time() - start_time}')
%%time

import requests

import multiprocessing

import time



session = None





def set_global_session():

    global session

    if not session:

        session = requests.Session()





def download_site(url):

    with session.get(url) as response:

        name = multiprocessing.current_process().name

        print(f"{name}: Read {len(response.content)} from {url}")





def download_all_sites(sites):

    with multiprocessing.Pool(initializer=set_global_session) as pool:

        pool.map(download_site, sites)





if __name__ == '__main__':

    sites = ["https://www.jython.org", "http://olympus.realpython.org/dice"]*80

    start_time = time.time()

    download_all_sites(sites)

    duration = time.time() - start_time

    print(f"Downloaded {len(sites)} in {duration} seconds")
%whois