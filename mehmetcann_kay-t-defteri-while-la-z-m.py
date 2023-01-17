#kayıt defteri kısa çözüm (döngülerle)

log=["00:01:34:23","43:14:06:46","00:16:06:48"]

x=0

while x<3:

    a,b,c,d=log[x].split(":")

    a=int(a)

    b=int(b)

    c=int(c)

    d=int(d)

    a=a*(60**2)*24

    b=b*(60**2)

    c=c*60

    x=x+1

    print(a+b+c+d)