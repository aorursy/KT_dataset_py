import re
text_string = '文本最重要的来源无疑是网络。我们要把网络中的文本获取形成一个文本数据库。利用一个爬虫抓取到网络中的信息。爬取的策略有广度爬取和深度爬取。根据用户的需求，爬虫可以有主题爬虫和通用爬虫之分。'
print(text_string)
p_string = text_string.split("。")
for line in p_string:
    print(line)
regex = "爬虫"
for line in p_string:
    match = re.search(regex,line)
    if(match is not None):
        print(match)
        print(line)
    
regex = "爬."
for line in p_string:
    if(re.search(regex,line) is not None):
        print(line)
regex = "文本"
for line in p_string:
    if(re.search(regex,line) is not None):
        print(line)
regex = "^文本"
for line in p_string:
    if(re.search(regex,line) is not None):
        print(line)
regex = "信息"
for line in p_string:
    if(re.search(regex,line) is not None):
        print(line)
regex = "信息$"
for line in p_string:
    if(re.search(regex,line) is not None):
        print(line)
text_string = ['[重要的]今年第七号台风23日登陆广东东部沿海地区','上海发布车库销售监管通知：违规者暂停网签资格','[紧要的]中国对印连发强硬信息，印度急切需要结束对峙']
regex = "^\[[重紧]..\]"
for line in text_string:
    if(re.search(regex,line) is not None):
        print(line)
strings = ['War of 1812', 'There are 5280 feet to a mile', 'Happy New Year 2016!']
regex = '[1-2][0-9]{3}'
for string in strings:
    if(re.search(regex,string) is not None):
        print(string)
years_string = '2016 was a good year, but 2017 will be better!'
regex = '[1-2][0-9]{3}'
ret = re.findall(regex,years_string)
print(ret)
