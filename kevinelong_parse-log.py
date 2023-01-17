lines_list = [
    "",
    "R1#",
    "*Feb 14 09:40:09.325: %LINK-3-UPDOWN: Interface GigabitEthernet0/1, changed state to down",
    "*Feb 14 09:40:10.326: %LINK-3-UPDOWN: Interface GigabitEthernet0/1, changed state to up",
    ""
]
                
# collet times that state change
# calculate difference between down and up

# print(lines_list)
down_time = None
up_time = None

for one_line in lines_list:
    if len(one_line) > 0:
        first_character = one_line[0]
#         print(first_character)
        if first_character == "*":
            parsed = one_line.split(": ")
#             print(parsed)
            date_time = parsed[0]
            item = parsed[1]
            details = parsed[2]
            print(details)
            words = details.split(" ")
            last_word = words[-1]
            print(last_word)
            if last_word == "down":
                down_time = date_time
            elif last_word == "up":
                up_time = date_time
                
import datetime
def fix_date_time(dt):
    parts = dt.split("*")
    text_date = parts[1]
#     Feb 14 09:40:09.325
    real_date_type = datetime.datetime.strptime(text_date, "%b %d %H:%M:%S.%f")
    fixed = real_date_type

    return fixed

up = fix_date_time(up_time)
down = fix_date_time(down_time) 
print(type(up))
print(type(down))
delta = up - down
print(type(delta))
print(delta)
# print(delta.__dir__())
print(delta.total_seconds() > 1.0)

