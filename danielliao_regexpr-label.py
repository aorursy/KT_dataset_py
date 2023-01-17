import re

str1 = 'data/oxford-iiit-pet/images/american_bulldog_146.jpg'

str2 = '/kaggle/input/images/images/basset_hound_188.jpg'

str3 = '/images/Siamese_83.jpg'

pat = r'([^/]+)_\d+.jpg$'

pat = re.compile(pat)

print(pat.search(str1).group(1))

print(pat.search(str2).group(1))

print(pat.search(str3).group(1))