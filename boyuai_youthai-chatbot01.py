import re

import random
_x = r'([1-9][0-9]*|0)'

_pattern = re.compile(_x)  

_match = _pattern.match('01')

print(_match)

print(_match.group(0))    
pairs = [

  (r'我觉得(.*)',                              #------------------------------重点(.*)

  ( "为什么你觉得%1?",                          #------------------------------解释%1

    "我不要你觉得，我要我觉得",

    "我也觉得%1.")), 



  (r'我好(.*)',

  ( "和我分享一下你为什么%1吧",

    "你好%1",

    "你是因为%1所以来找我聊天吗")),



  

  (r'(.*)[0-9](.*)',                                           

  ( "呜呜呜我数学不好",

    "看见数字我就头秃")),





  (r'(.*)\?',                                                   #------------------------------解释\?

  ( "为什么这么问?",

    "你自己再好好想想",

    "可能答案只有你自己才知道")),



  (r'(再见|拜拜)',                                             #------------------------------二选一

  ( "谢谢你和我聊天",

    "拜拜",

    "本次聊天收取费用100元，欢迎下次再来")),



  (r'(.*)',

  ( "我不明白你说了什么",

    "让我们换个话题，说说你的家人吧？",

    "为什么说%1?",

    "我明白了",

    "哈哈哈哈哈哈哈哈哈",

    "%1。",

    "好滴我知道啦",

    "那你感觉怎么样?",

    "我也觉得%1"))

]





pairs = [(re.compile(x, re.IGNORECASE),y) for (x,y) in pairs]   #------------------------re.compile
reflections = {

  "我"     : "你",

  "你"     : "我"

}
def substitute(str):                             #-----------------------------学生自己写

    words = ""

    for word in str:

        if word in reflections:

            word = reflections[word]

        words +=word

    return words

def wildcards( response, info):               #------------------------------find 和match

    pos = response.find('%')

    while pos >= 0:

        response = response[:pos] + substitute(info) + response[pos+2:]

    return response

def respond( str):                              #------------------.match()

    for (pattern, response) in pairs:

        match = pattern.match(str)

        if match:

            info = match.group(1)

            resp = random.choice(response)    # pick a random response Why do you say that %1?

            resp = wildcards(resp, info)     # process wildcards

            return resp

        

print('='*72)

print("二傻：你好！我是二傻！")

quit = ['再见','拜拜']

you = ''

while (you not in quit):         

    you = input('你：')                              

    print("二傻：" + respond(you)) 


