# Solution problem with Event Loop in Jupyter kernel https://github.com/jupyter/notebook/issues/3397#issuecomment-376803076



!pip install nest_asyncio



import nest_asyncio

nest_asyncio.apply()
# Install lib for asynchronous work with http.

!pip install aiohttp -qq
import asyncio

import aiohttp

import socket



from bs4 import BeautifulSoup





SEMA = asyncio.BoundedSemaphore(200)# 100 - 500 | the more the faster, and the greater the chance to get a ban.

####################### Get all PMID (page id: https://www.ncbi.nlm.nih.gov/pubmed/  -> 32361862 <-) #########################



async def get_session_and_inputs(page, items_per_page):



    headers = {"User-Agent" : "Mozilla/5.0", "Connection": "close"}

    async with aiohttp.ClientSession(headers=headers) as session:

        async with session.get(GETURL, verify_ssl=False) as response:

            data = await response.text()

            soup = BeautifulSoup(data, "lxml")



            inputs = {i['name']: i.get('value', '') for i

                      in soup.select('form#EntrezForm input[name]')}

            inputs['EntrezSystem2.PEntrez.PubMed.Pubmed_ResultsPanel.Pubmed_DisplayBar.PageSize'] = ITEMS_PER_PAGE

            inputs['EntrezSystem2.PEntrez.PubMed.Pubmed_ResultsPanel.Pubmed_DisplayBar.PrevPageSize'] = ITEMS_PER_PAGE

            inputs['EntrezSystem2.PEntrez.DbConnector.Cmd'] = 'PageChanged'



            inputs['EntrezSystem2.PEntrez.PubMed.Pubmed_ResultsPanel.Pubmed_Pager.CurrPage'] = page

            inputs['EntrezSystem2.PEntrez.PubMed.Pubmed_ResultsPanel.Pubmed_Pager.cPage'] = page



        async with session.post(POSTURL, data=inputs) as response2:

            data = await response2.text()

            soup = BeautifulSoup(data, "lxml")

            PMID.append([pmid.text for pmid in soup('dd')])





async def get_all_pmid():

    tasks = []

    for page in range(NUM_PAGES):

        task = asyncio.ensure_future(get_session_and_inputs(page=page,

                                                            items_per_page=NUM_PAGES))

        tasks.append(task)



    await asyncio.gather(*tasks)



############# Get all data in MAP from each uniq PMID articles (Title and Abstract) ###################

"""Example:

Title

Brain Recording, Mind-Reading, and Neurotechnology: Ethical Issues from Consumer Devices to Brain-Based Speech Decoding.



Abstract

Brain reading technologies are rapidly being developed in a number of neuroscience fields. 

...

...

...

etc""";





async def parse_text_from_paper(pmid):

    '''

    From link to science paper (pmid) - get html,

    then get document's body (data) from one page.

    '''

    

    conn = aiohttp.TCPConnector(family=socket.AF_INET)

    headers={"User-Agent" : "Mozilla/5.0",

             "Connection": "close"}

    

    async with aiohttp.ClientSession(headers=headers, connector=conn) as session:

        async with SEMA, session.get(f"https://www.ncbi.nlm.nih.gov/pubmed/{pmid}") as response:

            data = await response.text()

            soup = BeautifulSoup(data, "lxml")



            title = [text.find('h1').text for text in soup.find_all("div", class_="rprt_all")]

            data = [text.find('p').text for text in soup.find_all("div", class_="abstr")]

            pubmed_MAP[POSTURL+pmid] = data, title[0]

                

async def get_all_data_p(PMID):



    tasks = []

    pmid_items = [pmid_i for lst in PMID for pmid_i in lst]



    for pmid in pmid_items:



        task = asyncio.ensure_future(parse_text_from_paper(pmid))

        tasks.append(task)



    await asyncio.gather(*tasks)
def remove_none(data):

    print("Before removing nan:", len(data))

    filtered = {k: v for k, v in data.items() if v is not None}

    data.clear()

    data.update(filtered)

    print("After removing nan:", len(data))

    return data
import pandas as pd



# Initializing an empty dataframe

df = pd.DataFrame()
%%time



Topic = "Deep Learning"



GETURL = "https://www.ncbi.nlm.nih.gov/pubmed/?term=deep+learning"

POSTURL = "https://www.ncbi.nlm.nih.gov/pubmed/"



PMID = []

ITEMS_PER_PAGE = 100

NUM_PAGES = 133

pubmed_MAP = {}



loop = asyncio.get_event_loop()

loop.run_until_complete(get_all_pmid())

loop = asyncio.get_event_loop()

loop.run_until_complete(get_all_data_p(PMID))



print(f"{Topic} Done!\nMap length: {len(pubmed_MAP)}")
# Remove empty rows

pubmed_MAP = remove_none(pubmed_MAP)



# Write in dataframe

df["deep_learning"] = pubmed_MAP.values()

df["deep_learning_links"] = pubmed_MAP.keys()

df
%%time



Topic = "Covid 19"



GETURL = "https://www.ncbi.nlm.nih.gov/pubmed/?term=covid+19"

POSTURL = "https://www.ncbi.nlm.nih.gov/pubmed/"



PMID = []

ITEMS_PER_PAGE = 100

NUM_PAGES = 89

pubmed_MAP = {}



loop = asyncio.get_event_loop()

loop.run_until_complete(get_all_pmid())

loop = asyncio.get_event_loop()

loop.run_until_complete(get_all_data_p(PMID))



print(f"{Topic} Done!\nMap length: {len(pubmed_MAP)}")
# Remove empty rows

pubmed_MAP = remove_none(pubmed_MAP)



# Write in dataframe

df["covid_19"] = pd.Series(list(pubmed_MAP.values()))

df["covid_19_links"] = pd.Series(list(pubmed_MAP.keys()))

df
%%time



Topic = "Human Connectome"



GETURL = "https://www.ncbi.nlm.nih.gov/pubmed/?term=human+connectome"

POSTURL = "https://www.ncbi.nlm.nih.gov/pubmed/"



PMID = []

NUM_PAGES = 48

ITEMS_PER_PAGE = 100

pubmed_MAP = {}



loop = asyncio.get_event_loop()

loop.run_until_complete(get_all_pmid())

loop = asyncio.get_event_loop()

loop.run_until_complete(get_all_data_p(PMID))



print(f"{Topic} Done!\nMap length: {len(pubmed_MAP)}")
# Remove empty rows

pubmed_MAP = remove_none(pubmed_MAP)



# Write in dataframe

df["human_connectome"] = pd.Series(list(pubmed_MAP.values()))

df["human_connectome_links"] = pd.Series(list(pubmed_MAP.keys()))

df
%%time



Topic = "Virtual Reality"



GETURL = "https://www.ncbi.nlm.nih.gov/pubmed/?term=virtual+reality"

POSTURL = "https://www.ncbi.nlm.nih.gov/pubmed/"



PMID = []

ITEMS_PER_PAGE = 100

NUM_PAGES = 114

pubmed_MAP = {}



loop = asyncio.get_event_loop()

loop.run_until_complete(get_all_pmid())

loop = asyncio.get_event_loop()

loop.run_until_complete(get_all_data_p(PMID))



print(f"{Topic} Done!\nMap length: {len(pubmed_MAP)}")
# Remove empty rows

pubmed_MAP = remove_none(pubmed_MAP)



# Write in dataframe

df["virtual_reality"] = pd.Series(list(pubmed_MAP.values()))

df["virtual_reality_links"] = pd.Series(list(pubmed_MAP.keys()))

df
%%time



Topic = "Brain-Machine Interfaces"



GETURL = "https://www.ncbi.nlm.nih.gov/pubmed/?term=Brain-Machine+Interfaces"

POSTURL = "https://www.ncbi.nlm.nih.gov/pubmed/"



PMID = []

ITEMS_PER_PAGE = 100

NUM_PAGES = 43

pubmed_MAP = {}



loop = asyncio.get_event_loop()

loop.run_until_complete(get_all_pmid())

loop = asyncio.get_event_loop()

loop.run_until_complete(get_all_data_p(PMID))



print(f"{Topic} Done!\nMap length: {len(pubmed_MAP)}")
# Remove empty rows

pubmed_MAP = remove_none(pubmed_MAP)



# Write in dataframe

df["brain_machine_interfaces"] = pd.Series(list(pubmed_MAP.values()))

df["brain_machine_interfaces_links"] = pd.Series(list(pubmed_MAP.keys()))

df
%%time



Topic = "Electroactive Polymers"



GETURL = "https://www.ncbi.nlm.nih.gov/pubmed/?term=electroactive+Polymers"

POSTURL = "https://www.ncbi.nlm.nih.gov/pubmed/"



PMID = []

ITEMS_PER_PAGE = 100

NUM_PAGES = 10

pubmed_MAP = {}



loop = asyncio.get_event_loop()

loop.run_until_complete(get_all_pmid())

loop = asyncio.get_event_loop()

loop.run_until_complete(get_all_data_p(PMID))



print(f"{Topic} Done!\nMap length: {len(pubmed_MAP)}")
# Remove empty rows

pubmed_MAP = remove_none(pubmed_MAP)



# Write in dataframe

df["electroactive_polymers"] = pd.Series(list(pubmed_MAP.values()))

df["electroactive_polymers_links"] = pd.Series(list(pubmed_MAP.keys()))

df
%%time



Topic = "PEDOT electrodes"



GETURL = "https://www.ncbi.nlm.nih.gov/pubmed/?term=PEDOT+electrodes"

POSTURL = "https://www.ncbi.nlm.nih.gov/pubmed/"



PMID = []

ITEMS_PER_PAGE = 100

NUM_PAGES = 8

pubmed_MAP = {}



loop = asyncio.get_event_loop()

loop.run_until_complete(get_all_pmid())

loop = asyncio.get_event_loop()

loop.run_until_complete(get_all_data_p(PMID))



print(f"{Topic} Done!\nMap length: {len(pubmed_MAP)}")
# Remove empty rows

pubmed_MAP = remove_none(pubmed_MAP)



# Write in dataframe

df["pedot_electrodes"] = pd.Series(list(pubmed_MAP.values()))

df["pedot_electrodes_links"] = pd.Series(list(pubmed_MAP.keys()))

df
Topic = "Neuroprosthetics"



GETURL = "https://www.ncbi.nlm.nih.gov/pubmed/?term=neuroprosthetics"

POSTURL = "https://www.ncbi.nlm.nih.gov/pubmed/"



PMID = []

ITEMS_PER_PAGE = 100

NUM_PAGES = 8

pubmed_MAP = {}



loop = asyncio.get_event_loop()

loop.run_until_complete(get_all_pmid())

loop = asyncio.get_event_loop()

loop.run_until_complete(get_all_data_p(PMID))



print(f"{Topic} Done!\nMap length: {len(pubmed_MAP)}")
# Remove empty rows

pubmed_MAP = remove_none(pubmed_MAP)



# Write in dataframe

df["neuroprosthetics"] = pd.Series(list(pubmed_MAP.values()))

df["neuroprosthetics_links"] = pd.Series(list(pubmed_MAP.keys()))

df
# Sorted

cols = df.columns.tolist()

df = df[cols[::2] + cols[1::2]]



# Save to CSV

df.to_csv('pubmed_abstracts.csv')

df