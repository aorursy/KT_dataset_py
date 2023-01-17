from datetime import datetime

from collections import defaultdict, Counter



all_events = defaultdict(list)

all_shops = set()



with open('order_brush_order.csv', 'r') as f:

  for line in f.readlines()[1:]:

    orderid, shopid, userid, event_time = line.strip().split(',')

    year, month, event_time = event_time.strip().split('-')

    day, event_time = event_time.split(' ')

    hour, minute, second = event_time.strip().split(':')

    orderid, shopid, userid, year, month, day, hour, minute, second

    year, month, day, hour, minute, second = map(int, (year, month, day, hour, minute, second))

    dt = datetime(year=year, month=month, day=day, hour=hour, minute=minute, second=second)

    shopid = int(shopid)

    all_events[shopid].append((dt, int(userid), int(orderid)))

    all_shops.add(shopid)



for shopid, events in all_events.items():

  events.sort()



brush = defaultdict(set)

brush_pts = defaultdict(set)



for shopid, events in all_events.items():

  i = 0

  while i < len(events):

    order = 0

    user = defaultdict(int)



    for j in range(i, len(events)):

      secs = (events[j][0] - events[i][0]).total_seconds()



      if secs > 60 * 60:

        break



      userid = events[j][1]

      order += 1

      user[userid] += 1



      if j + 1 < len(events) and events[j + 1][0] == events[j][0]:

        # only compute score once this time is done

        continue



      if order >= 3 * len(user):

        brush_pts[shopid] = brush_pts[shopid].union(set(range(i, j + 1)))

    

    while i + 1 < len(events) and events[i + 1][0] == events[i][0]:

      i += 1



    i += 1



with open('result.csv', 'w') as f:

  f.write('shopid,userid\n')



  for shopid in sorted(all_shops, key=int):

    if shopid not in brush_pts:

      f.write(str(shopid) + ',0\n')

    else:

      freq = Counter()

      events = all_events[shopid]

      for i in brush_pts[shopid]:

        freq[events[i][1]] += 1



      items = sorted(freq.items(), key=lambda tup: tup[1])



      ids = sorted([str(item[0]) for item in items if item[1] == items[-1][1]], key=int)



      f.write(str(shopid) +','+ '&'.join(ids) + '\n')
