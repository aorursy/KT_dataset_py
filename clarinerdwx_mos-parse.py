import requests

from bs4 import BeautifulSoup

import pandas as pd

from urllib3 import disable_warnings, exceptions

import re



disable_warnings(exceptions.InsecureRequestWarning)
def get_mos(station: str, model: str):

    """

    Gets the MOS Data from NOAA. Models are GFS, NAM, and GFS Extended (GFSX).

    

    Passes parameter sta: station.

    Return HTML file for parsing

    """

    params = {"sta": station}

    base_url = "https://www.nws.noaa.gov/cgi-bin/mos"

    if model.upper() == "GFS":

        url = f"{base_url}/getmav.pl"

    elif model.upper() == "NAM":

        url = f"{base_url}/getmet.pl"

    elif model.upper() == "GFSX":

        url = f"{base_url}/getmex.pl"

    response = None

    try:

        r = requests.get(url=url, params=params, verify=False)

        r.raise_for_status()

        response = r.text

    except requests.HTTPError as http_err:

        print(http_err)

    except Exception as e:

        print(e)

    finally:

        return response
kash_mos = get_mos(station="KASH", model="GFS")

kash_mos
kash_data = BeautifulSoup(kash_mos, "html.parser")

data = [string for string in kash_data.stripped_strings][2]

data.split("\n")  # This is just so that you can read the contents of data, the real string is what we'll be parsing.
x = data.split()

# for integration into Daviology38/python_MOS

# at this point, all future code will be based on Dave's script
x[:7]