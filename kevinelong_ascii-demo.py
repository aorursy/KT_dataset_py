def numbers(v):
    print(v, bin(v), hex(v), chr(v))


def show_ascii(name):
    for letter in name:
        ascii_code = ord(letter)
        numbers(ascii_code)


show_ascii("KEVIN")

r = 200
g = 240
b = 64

rgb = [r, g, b]

for v in rgb:
    numbers(v)

for n in range(0,256):
    print(n, chr(n), end="\t")

# start = int("1F600", 16) # convert from hex string e.g. decimal 128512 
# stop = int("1F64F", 16) # convert from hex string e.g decimal 128591
start = ord("😀")
stop = ord("🙏")
for i in range(start,stop+1):
    print(hex(i),chr(i), end="\t")
print(type(""))
data = [u'host_name', u'ipv4_addr', u'net_mask', u'vlan_num', u'vrrp', u'v6_addr'] 
for d in data:
    print(len(d), d, type(d))