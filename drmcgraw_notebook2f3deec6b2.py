import time
timestamp = "2008-09-26T01:51:42.000Z"
ts = time.strftime("%Y-%m-%d %H:%M:%S",time.strptime(timestamp[:19], "%Y-%m-%dT%H:%M:%S"))
ts = time.strptime(ts, "%Y-%m-%d %H:%M:%S")
print (ts)


