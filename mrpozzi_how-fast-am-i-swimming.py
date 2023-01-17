import datetime





def strfdelta(tdelta, fmt):

    """

    https://stackoverflow.com/questions/8906926/formatting-python-timedelta-objects

    """

    total_seconds = int(tdelta.total_seconds())

    hours, remainder = divmod(total_seconds,3600)

    minutes, seconds = divmod(remainder,60)

    delta = {'hours': hours, 'minutes': minutes, 'seconds': seconds}

    return fmt.format(**delta)





def hundred_split_to_total(minutes, seconds, num_hundreds):

    total = num_hundreds * datetime.timedelta(minutes=minutes, seconds=seconds)

    print(strfdelta(total, "{hours}:{minutes}:{seconds}"))





def total_to_hundred_split(hours, minutes, seconds, num_hundreds):

    total = datetime.timedelta(hours=hours, minutes=minutes, seconds=seconds) / num_hundreds

    print(strfdelta(total, "{hours}:{minutes}:{seconds}"))





def hundred_pace_to_mile_pace(minutes, seconds):

    total = 16.5 * datetime.timedelta(minutes=minutes, seconds=seconds)

    miles_h = datetime.timedelta(minutes=60) / total

    print("{miles_h} miles/h".format(miles_h=miles_h))





def total_to_mph(hours, minutes, seconds, num_miles):

    print(num_miles * datetime.timedelta(hours=1)/datetime.timedelta(hours=hours, minutes=minutes, seconds=seconds))
# mile pace

hundred_split_to_total(1, 50, 16.5)

hundred_split_to_total(1, 35, 16.5)

hundred_split_to_total(1, 40, 16.5)
# miles per hour

hundred_pace_to_mile_pace(1, 50)

hundred_pace_to_mile_pace(1, 35)

hundred_pace_to_mile_pace(1, 40)
total_to_hundred_split(10, 31, 0, 21.5*16.5)

total_to_mph(10, 31, 0, 21.5)
