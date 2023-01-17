class photo:

    def __init__(self, id = None,value = None ,tags = None,state = None ,m = None):

        self.id = id

        self.value = value;

        self.tags = tags

        self.state = state

        self.m = m

        
photos = []

with open('../input/hashcode-photo-slideshow/d_pet_pictures.txt', 'r') as f:

    lines = f.readlines()

total_photos = int(lines[0])

print(total_photos)

hp = []

vp = []

h = 0

v = 0

for i in range(total_photos):

    p = photo()

    p.id = i

    line = lines[i+1].split()

    if line[0] == 'H':

        p.state = 1

    else:

        p.state = 2

    p.m = int(line[1])

    p.tags = []

    value = 0

    for j in range(p.m):

        s = line[2+j]

        p.tags.append(s)

        val1 = 0

        for k in range (len(s)):

            val1+=max(0,(1<<(k%32))*ord(s[k]))

        if j==0:

            value = val1

        else:

            value = max(value,val1)

    p.value=value;

    if p.state == 1:

        h += 1

        hp.append(p)

    else:

        v += 1

        vp.append(p)

    photos.append(p)

ans=h+(v+1)//2

def get_value(photo):

    return photo.value

vp.sort(reverse = True,key = get_value)

hp.sort(reverse = True,key = get_value)

with open('submission.txt', 'w+') as opf:

    opf.write(str(ans)+'\n')

    for i in range(len(hp)):

        opf.write(str(hp[i].id)+'\n')

    for i in range(0,len(vp),2):

        opf.write(str(vp[i].id)+' '+str(vp[i+1].id)+'\n')