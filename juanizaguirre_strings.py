our_string = "Hello World!"
our_string
"I said: " + our_string
our_string * 3
our_string[2:5]
our_string[-5:-2]
our_string[1:]
our_string[:-1]

our_string[:5] = 'Howdy'
our_string[1]
our_string[0]
our_string[1]
our_string[2]
our_string[3]
our_string[4]
our_string[-1]
our_string.upper()
our_string.lower()
our_string.startswith('Hello')
our_string.endswith('World!')
our_string.endswith('world!')  # Python is case-sensitive
our_string.replace('World', 'there')
our_string.replace('o', '@', 1)   # only replace one o
'  hello 123  '.lstrip()    # left strip
'  hello 123  '.rstrip()    # right strip
'  hello 123  '.strip()     # strip from both sides
'  hello abc'.rstrip('cb')  # strip c's and b's from right
our_string.ljust(30, '-')
our_string.rjust(30, '-')
our_string.center(30, '-')
our_string.count('o')   # it contains two o's
our_string.index('o')   # the first o is our_string[4]
our_string.rindex('o')  # the last o is our_string[7]
'-'.join(['hello', 'world', 'test'])
'hello-world-test'.split('-')
our_string.upper()[3:].startswith('LO WOR')  # combining multiple things