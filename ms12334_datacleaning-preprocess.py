# import necessary public library

import re
def re_repl_count(pattern,replacement_texts,line):

    count = 0

    tmp = re.findall(pattern,line)

    count = len(tmp)

    result_str = re.sub(pattern,replacement_texts,line) 

    return result_str, count;
with open('reformatted.txt','w') as fout:

    with open('/kaggle/input/phylotree/mtDNA tree Build 17.htm','r',encoding='latin-1') as stream:

        contents = str(stream.read()).replace('\n',' ')

        contents, count1 = re_repl_count(r'</tr>\s+<tr ',r'</tr>\n<tr ',contents)

        contents, count1 = re_repl_count(r'</tr>\s+(<!\[if support)',r'</tr>\n\1',contents)

        fout.write(contents)
with open('result.csv', 'w') as fout:

    isTree = False

    """

    font517826  - state = @M~ - black

    font617826  - state = @m~ - black italic

    font917826  - state = @N~ - blue

    font1017826 - state = @n~ - blue italic

    """

    state = 0

    row_count = 0

    with open('reformatted.txt','r') as fin:

        for line in fin:

            if re.search(r'<td [^>]*?>L0</td>',line):

                isTree = True

            if re.search(r'<!\[if support',line):

                isTree = False

            if isTree:

                row_count += 1

                tmp, count = re_repl_count(r'</?span[^>]*?>',r'',line)

                tmp, count = re_repl_count(r'>\s+',r'>',tmp)

                tmp, count = re_repl_count(r'\s+<',r'<',tmp)

                tmp, count = re_repl_count(r'<td [^>]*?>&nbsp;</td>',r'_',tmp)

                tmp, count = re_repl_count(r'<tr [^>]*?>',r'',tmp)

                

                tmp, count = re_repl_count(r'<td [^>]*?><a></a></td>',r'',tmp)   

                tmp, count = re_repl_count(r'<td [^>]*?><a\s+target="_blank"\s+href="([^"]*?)">([^<]*?)</a></td>',r';\1;\2',tmp)

                tmp, count = re_repl_count(r's+;http',r';http',tmp)

                tmp, count = re_repl_count(r'_+</tr>',r'</tr>',tmp)

                tmp, count = re_repl_count(r'_+;http',r';http',tmp)

                               

                tmp, count = re_repl_count(r'<font\s+class="font517826">',r'@M~',tmp)

                tmp, count = re_repl_count(r'<font\s+class="font617826">',r'@m~',tmp)

                tmp, count = re_repl_count(r'<font\s+class="font917826">',r'@N~',tmp)

                tmp, count = re_repl_count(r'<font\s+class="font1017826">',r'@n~',tmp)

                tmp, count = re_repl_count(r'</font>',r'%',tmp)



                # for gene entry with label, remove the closing tag for the label

                tmp, count = re_repl_count(r'</td>(<td class[^>]*?>)',r'\1',tmp)

                tmp, count = re_repl_count(r'(<td class[^>]*?>)',r';\1',tmp)

                tmp, count = re_repl_count(r'(<td class[^>]*?>)(?=[^@])',r'\1@M~',tmp)

                tmp, count = re_repl_count(r'(?<=[^%>;])@',r'%@',tmp)

                tmp, count = re_repl_count(r'(?<=[^%])</td>',r'%',tmp)

                tmp, count = re_repl_count(r'<td class[^>]*?>',r'',tmp)

                # Remove closing tag for gene entry

                tmp, count = re_repl_count(r'</td>',r'',tmp)

                

                tmp, col_count = re_repl_count(r'_',r'',tmp)

                if re.search(r'<td rowspan[^>]*?>',tmp):

                    col_count = col_count + 1

                tmp, count = re_repl_count(r'<td rowspan[^>]*?>',r'',tmp)

                

                tmp, count = re_repl_count(r'\s+',r' ',tmp)

                tmp, count = re_repl_count(r'@.~%',r'',tmp)

                tmp, count = re_repl_count(r'</tr>',r'\n',tmp) 

                

                count = 1

                while (count > 0):

                    tmp, count = re_repl_count(r'(@.~)([^ %]+)\s',r'\1\2%\1',tmp)

                

                fout.write(str(row_count) + ';' + str(col_count) + ';' + tmp)

                col_count = 0
with open('result_newick.txt','w') as fout:

    with open('result.csv', 'r') as fin:

        depth = 0

        final = ''

        col_num = 0

        for line in fin:

            if re.match('[0-9]+;0;',line) is None:

                # get a column number

                col_num, count = re_repl_count(r'[0-9]+;([0-9]+);.+\n',r'\1',line)

                col_num = col_num.rstrip("\n")

                # get a label

                s, count = re_repl_count(r'[0-9]+;[0-9]+;(.*?)\n',r'\1',line)

                # escape exisinting single quotes in labels

                s, count = re_repl_count(r"'",r"''",s)

                

                # replace parenthesis in gene description with a period

                s, count = re_repl_count(r"\(",r".",s)

                s, count = re_repl_count(r"\)",r".",s)

                

                s, count = re_repl_count(r"@M~([^%]+)%",r"<<\1>>",s)

                s, count = re_repl_count(r"@m~([^%]+)%",r"<\1>",s)

                s, count = re_repl_count(r"@N~([^%]+)%",r"<<<<\1>>>>",s)

                s, count = re_repl_count(r"@n~([^%]+)%",r"<<<\1>>>",s)

                

                # colon used URL causes a problem in Newick format. So replace it with _

                s, count = re_repl_count(r'http:([^;]+);([^;]+);http:([^;]+);([^;]+)',r'.\2"http_\1"|\4"http_\3".',s)

                s, count = re_repl_count(r'http:([^;]+);([^;]+)',r'.\2"http_\1".',s)

                

                s, count = re_repl_count(r';',r'.',s)

                # enclose entire string with single quotes.

                s = "'" + s + "'"

                

                if col_num:

                    if depth < int(col_num):

                        diff = int(col_num) - depth

                        final = final + r'(' * diff + s

                    if depth == int(col_num):

                        final = final + r',' + s

                    if depth > int(col_num):

                        diff = depth - int(col_num)

                        final = final +  r')' * diff + s

                    depth = int(col_num)

    final = final + r')' * depth + ';'

    fout.write(final)