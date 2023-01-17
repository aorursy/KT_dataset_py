import pandas as pd
import numpy as np
import urllib.request
import urllib3
from urllib.request import urlopen
import urllib.parse
import re
import certifi
url = 'https://training.sap.com'
#HTTPConnection.request(read, url, body=None, headers={}, encode_chunked=False)
values={'s':'basics', 'd':'SAP HANA'}
data=urllib.parse.urlencode(values)
data1=data.encode('utf-8')
req=urllib.request.Request(url,data1)
#urlopen(req, context=ssl._create_unverified_context())
resp = urllib.request.urlopen((req))
respData=resp.read()
print(respData)
import http.client
import urllib.request
import http.client
conn = http.client.HTTPSConnection("www.python.org")
conn.request("HEAD", "/")
res = conn.getresponse()
print(res.status, res.reason)
data = res.read()
print(len(data))

import http.client

connection = http.client.HTTPConnection('www.python.org', 80, timeout=50)
print(connection)
