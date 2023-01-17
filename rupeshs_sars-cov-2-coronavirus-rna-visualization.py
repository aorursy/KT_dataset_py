from PIL import Image



width=173

height=173



im_out = Image.new("RGB", (width, height), "white") #new RGB image to hold the result

rgb_im_out = im_out.convert('RGB')

count=0

sequences=""

with open("/kaggle/input/sars-cov2-rna-sequence-wuhan/sequences.fasta") as f:

    next(f)

    for line in f:

        count=count+1

        for v in line:

            if v.isspace()==False:

               sequences=sequences+v

   

print("Length of RNA sequence :" +str(len(sequences)))

color_A=(0,128,255)

color_G=(0, 255, 0)

color_T=(248,222,126)

color_C=(255,0,0)

seq_index=0



for y in range(0,height):

    for x in range (0,width):

        #print(seq_index)

        if  seq_index>=len(sequences):

            rgb_im_out.putpixel((x,y),(255,255,255))

        elif sequences[seq_index]=="A":

            rgb_im_out.putpixel((x,y),color_A)

        elif sequences[seq_index]=="G":

            rgb_im_out.putpixel((x,y),color_G)

        elif sequences[seq_index]=="T":

            rgb_im_out.putpixel((x,y),color_T)

        elif sequences[seq_index]=="C":

            rgb_im_out.putpixel((x,y),color_C)

        seq_index=seq_index+1 

        

rgb_im_out = rgb_im_out.resize((width*4,height*4)) 

    

rgb_im_out.save("rnaseq_color.png")
rgb_im_out
rgb_im_out = rgb_im_out.resize((width,height)) 



color_A=(128,128,128)

color_G=(211,211,211)

color_T=(192,192,192)

color_C=(169,169,169)



seq_index=0

for y in range(0,height):

    for x in range (0,width):

        #print(seq_index)

        if  seq_index>=len(sequences):

            rgb_im_out.putpixel((x,y),(255,255,255))

        elif sequences[seq_index]=="A":

            rgb_im_out.putpixel((x,y),color_A)

        elif sequences[seq_index]=="G":

            rgb_im_out.putpixel((x,y),color_G)

        elif sequences[seq_index]=="T":

            rgb_im_out.putpixel((x,y),color_T)

        elif sequences[seq_index]=="C":

            rgb_im_out.putpixel((x,y),color_C)

        seq_index=seq_index+1 

        

rgb_im_out = rgb_im_out.resize((width*4,height*4)) 

    

rgb_im_out.save("rnaseq_gray.png")
rgb_im_out