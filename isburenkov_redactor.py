def нашлось(v):

    return find(v)
нашлось(2)
txt = "Hello, welcome to my world."



x = txt.find("welcome")



print(x) 
def нашлось(строка, v):

    return строка.find(v)
нашлось("12345", "3")
def нашлось(строка, v):

    if строка.find(v) == -1:

        return "ложь"

    else:

        return "истина"
нашлось("12345", "3")
нашлось("12345", "7")
строка = "8" * 70
строка
def нашлось(v):

    global строка

    if строка.find(v) == -1:

        return "ложь"

    else:

        return "истина"
нашлось("2")
нашлось("8")
нашлось(8888)
def нашлось(v):

    global строка

    v = str(v) # явно превращаем число в строку

    if строка.find(v) == -1:

        return "ложь"

    else:

        return "истина"

нашлось(8888)
нашлось(2222)