"""

import requests

from requests.adapters import HTTPAdapter

from requests.packages.urllib3.util.retry import Retry



headers1={'user-key' : zomato_api_key}

read_rest=[]

rest_list=[]



def call_1(): 

    for i in range(1, 80):

        read_rest=[]

        try:

            rests_url=('https://developers.zomato.com/api/v2.1/search?entity_id='+str(i)+'&entity_type=city&start=101&count=20') # fetching the data from this url    

            get_request = requests.get(rests_url, headers=headers1)

            read_rest=json.loads(get_request.text) #loading the data fetched to an object

            rest_list.append(read_rest)

        except requests.exceptions.ConnectionError as r:

            r.status_code = "Connection refused"



"""
#!/usr/bin/env bash



## REMOVE THE COMMENTS(#)FROM ALL THE LINES AFTER THIS!





#SHELL=/bin/sh

#PATH=PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/home/akashram/



#0 0 * * SUN <path_to_file>  - /home/email_id/shell_script_filename